import logging
import re
import numpy as np
from sqlalchemy import select, or_
from flashrank import Ranker, RerankRequest

from app.db.database import AsyncSessionLocal
from app.models import ReviewRule, Technology
from app.services.ai import (
    analyze_code_with_ai,
    generate_embedding,
    validate_user_message,
    generate_chat_reply,
)
from app.services.github import (
    get_installation_access_token,
    get_repo_file,
    get_pr_diff,
    get_pr_comments,
    post_review_to_github,
    post_reply_to_github,
)

logger = logging.getLogger("ai_reviewer.services.review")

try:
    ranker = Ranker(cache_dir="/tmp/models")
    logger.info("Ranker model initialized successfully.")
except Exception:
    logger.error("Failed to initialize Ranker model.", exc_info=True)
    ranker = None


def parse_requirements(req_text: str) -> list[str]:
    if not req_text:
        return []
    techs = set()
    for line in req_text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            match = re.match(r"^([a-zA-Z0-9_\-]+)", line)
            if match:
                techs.add(match.group(1).lower())
    return list(techs)


async def process_pull_request(payload: dict) -> None:
    installation_id_raw = payload.get("installation", {}).get("id")
    if not installation_id_raw:
        logger.warning("Missing installation ID in GitHub payload.")
        return

    installation_id = int(installation_id_raw)
    pr_data = payload.get("pull_request", {})
    pr_number = int(pr_data.get("number", 0))

    try:
        token = str(get_installation_access_token(installation_id))
        repo_full_name = str(payload.get("repository", {}).get("full_name", ""))
        head_sha = str(pr_data.get("head", {}).get("sha", ""))

        logger.info(f"Initiating PR review for {repo_full_name}#{pr_number}")

        diff_text = await get_pr_diff(repo_full_name, pr_number, token)
        if not diff_text:
            logger.info(f"PR #{pr_number} contains no diff data. Aborting review.")
            return

        diff_vector = await generate_embedding(diff_text)

        async with AsyncSessionLocal() as db:
            req_text = await get_repo_file(
                repo_full_name, "requirements.txt", head_sha, token
            )
            tech_names = parse_requirements(req_text)

            conditions = [ReviewRule.is_global.is_(True)]
            if tech_names:
                conditions.append(Technology.name.in_(tech_names))

            stmt = (
                select(ReviewRule)
                .outerjoin(ReviewRule.technologies)
                .where(ReviewRule.is_active.is_(True), or_(*conditions))
            )

            if diff_vector:
                stmt = stmt.order_by(
                    ReviewRule.embedding.cosine_distance(diff_vector)
                ).limit(50)
            else:
                stmt = stmt.limit(50)

            result = await db.execute(stmt)
            raw_rules = list(result.scalars().all())

            final_rules = []
            if raw_rules and ranker and diff_text:
                passages = [
                    {
                        "id": str(r.id),
                        "text": f"{r.name}\nBad: {r.example_bad}\nGood: {r.example_good}",
                    }
                    for r in raw_rules
                ]
                rank_request = RerankRequest(query=diff_text, passages=passages)
                reranked_results = ranker.rerank(rank_request)

                if len(reranked_results) <= 2:
                    final_rule_ids = [int(item["id"]) for item in reranked_results]
                    final_rules = [r for r in raw_rules if r.id in final_rule_ids]
                else:
                    scores = [float(item["score"]) for item in reranked_results]
                    adaptive_threshold = float(np.mean(scores)) + (
                        0.5 * float(np.std(scores))
                    )

                    final_rule_ids = []
                    for i in range(len(reranked_results)):
                        current_score = float(reranked_results[i]["score"])
                        if current_score >= adaptive_threshold:
                            final_rule_ids.append(int(reranked_results[i]["id"]))
                            continue

                        if i > 0:
                            previous_score = float(reranked_results[i - 1]["score"])
                            if (
                                previous_score > 0
                                and (previous_score - current_score) / previous_score
                                > 0.5
                            ):
                                break

                        final_rule_ids.append(int(reranked_results[i]["id"]))

                    final_rules = [r for r in raw_rules if r.id in final_rule_ids]
            else:
                final_rules = raw_rules[:5]

            raw_comments = await analyze_code_with_ai(diff_text, final_rules)

            grouped = {}
            for c in raw_comments:
                key = (str(c.get("file_path", "")), int(c.get("line_number", 0)))
                if key not in grouped:
                    grouped[key] = {
                        "file_path": c.get("file_path"),
                        "line_number": c.get("line_number"),
                        "rule_id": str(c.get("rule_id", "0")),
                        "explanation": str(c.get("explanation", "")),
                        "suggested_code": str(c.get("suggested_code", "")),
                        "is_inline_fix": bool(c.get("is_inline_fix", False)),
                    }
                else:
                    grouped[key]["rule_id"] += f", {str(c.get('rule_id', '0'))}"
                    grouped[key]["explanation"] += (
                        f"\n\n---\n**Крім того:**\n{str(c.get('explanation', ''))}"
                    )

            existing = await get_pr_comments(repo_full_name, pr_number, token)
            signatures = set()

            for ec in existing:
                if ec.get("user", {}).get("type") == "Bot":
                    comment_line = ec.get("line") or ec.get("original_line")
                    path = ec.get("path")
                    match = re.search(
                        r"<!-- rule_ids: (.+?) -->", str(ec.get("body", ""))
                    )
                    if comment_line and path and match:
                        rule_ids = match.group(1).split(",")
                        for rid in rule_ids:
                            signatures.add(
                                (str(path), int(comment_line), str(rid).strip())
                            )

            final_comments = []
            for c in grouped.values():
                file_path = str(c.get("file_path", ""))
                line = int(c.get("line_number", 0))
                rule_ids_str = str(c.get("rule_id", "0"))
                current_rule_ids = [r.strip() for r in rule_ids_str.split(",")]

                explanation = c.get("explanation", "").strip()
                suggested_code = c.get("suggested_code", "").strip()
                is_inline = c.get("is_inline_fix", False)

                if is_inline and suggested_code:
                    safe_comment = (
                        f"{explanation}\n\n```suggestion\n{suggested_code}\n```"
                    )
                elif suggested_code:
                    safe_comment = f"{explanation}\n\n```python\n{suggested_code}\n```"
                else:
                    safe_comment = explanation

                is_duplicate = False
                for r in current_rule_ids:
                    for offset in range(-5, 6):
                        if (file_path, line + offset, r) in signatures:
                            is_duplicate = True
                            break
                    if is_duplicate:
                        break

                if not is_duplicate:
                    c["comment"] = (
                        f"{safe_comment}\n\n<!-- rule_ids: {rule_ids_str} -->"
                    )
                    final_comments.append(c)

            if final_comments:
                await post_review_to_github(
                    repo_full_name, pr_number, head_sha, final_comments, token
                )

    except Exception:
        logger.error(f"Pipeline execution failed for PR #{pr_number}", exc_info=True)


async def process_review_comment(payload: dict) -> None:
    comment = payload.get("comment", {})
    if comment.get("user", {}).get("type") == "Bot":
        return

    installation_id_raw = payload.get("installation", {}).get("id")
    if not installation_id_raw:
        return

    installation_id = int(installation_id_raw)
    comment_id = int(comment.get("id", 0))

    try:
        token = str(get_installation_access_token(installation_id))
        repo_full_name = str(payload.get("repository", {}).get("full_name", ""))
        pr_number = int(payload.get("pull_request", {}).get("number", 0))
        user_text = str(comment.get("body", ""))
        diff_hunk = str(comment.get("diff_hunk", ""))

        validation = await validate_user_message(user_text)
        if not validation.get("is_safe"):
            await post_reply_to_github(
                repo_full_name,
                pr_number,
                comment_id,
                f"Відхилено: {str(validation.get('reason', ''))}",
                token,
            )
            return

        reply = await generate_chat_reply(diff_hunk, user_text)
        await post_reply_to_github(
            repo_full_name, pr_number, comment_id, str(reply), token
        )

    except Exception:
        logger.error(
            f"Failed to resolve review thread for comment {comment_id}", exc_info=True
        )
