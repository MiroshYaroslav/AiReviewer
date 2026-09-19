import json
import re
import logging
from google import genai
from groq import AsyncGroq
from app.core.config import settings
from app.models import ReviewRule

logger = logging.getLogger("ai_reviewer.services.ai")

GEMINI_API_KEYS = settings.gemini_keys_list
GROQ_API_KEYS = settings.groq_keys_list

current_gemini_idx = 0
current_groq_idx = 0


def prepare_numbered_diff(diff_text: str) -> str:
    lines = diff_text.split("\n")
    numbered_diff = []
    current_line = 0

    for line in lines:
        if line.startswith("@@"):
            match = re.search(r"\+(\d+)(?:,\d+)?", line)
            if match:
                current_line = int(match.group(1))
            numbered_diff.append(line)
        elif line.startswith("+") and not line.startswith("+++"):
            numbered_diff.append(f"[{current_line}] {line}")
            current_line += 1
        elif line.startswith(" ") or (line == "" and current_line > 0):
            numbered_diff.append(f"[{current_line}] {line}")
            current_line += 1
        else:
            numbered_diff.append(line)
    return "\n".join(numbered_diff)


async def analyze_code_with_ai(diff_text: str, rules: list[ReviewRule]) -> list[dict]:
    global current_groq_idx
    numbered_diff = prepare_numbered_diff(diff_text)

    rules_text = "\n".join(
        [
            f"[R{r.id}] {r.name}\nBad: {r.example_bad}\nGood: {r.example_good}\n"
            for r in rules
        ]
    )

    prompt = f"""You are a strict Senior Security & Architecture Reviewer. Your task is to conduct an EXHAUSTIVE, line-by-line analysis of the provided code diff.

    [AVAILABLE RULES]
    {rules_text}

    [CODE DIFF]
    {numbered_diff}

    CRITICAL INSTRUCTIONS (OBLIGATORY):
    1. ALGORITHMIC CHECK: You MUST analyze EVERY SINGLE LINE of the diff starting with a '+'.
    2. HONESTY BY DESIGN: Do not fabricate issues. If a line does not violate any rule exactly, explicitly ignore it.
    3. EXACT INDENTATION (CRITICAL): Python relies on indentation. Your "suggested_code" MUST contain the EXACT SAME leading spaces as the original code line. Do NOT strip leading whitespace.
    4. STRICT JSON ESCAPING: You MUST properly escape all double quotes (\\") and newlines (\\n) inside your JSON string values.

    OUTPUT FORMAT:
    You MUST structure your response in TWO sequential phases.

    PHASE 1: THE SCRATCHPAD
    You MUST open a `<scratchpad>` tag. Inside, write down your thought process for EVERY added line.

    PHASE 2: THE STRICT JSON ARRAY
    ONLY AFTER closing `</scratchpad>`, generate a strict JSON array containing the confirmed violations. Do NOT wrap the JSON in markdown blocks like ```json.
    Each JSON object MUST have EXACTLY these keys:
    - "file_path": string
    - "line_number": integer
    - "rule_id": string
    - "rule_name": string
    - "current_code": string
    - "explanation": string
    - "suggested_code": string (MUST preserve original leading spaces)
    - "is_inline_fix": boolean
    """

    attempts = 0
    max_attempts = len(GROQ_API_KEYS) * 2

    while attempts < max_attempts:
        client = AsyncGroq(api_key=GROQ_API_KEYS[current_groq_idx])
        try:
            # noinspection PyTypeChecker
            response = await client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw_text = (response.choices[0].message.content or "").strip()

            json_section = (
                raw_text.split("</scratchpad>")[-1]
                if "</scratchpad>" in raw_text
                else raw_text
            )
            json_section = (
                json_section.replace("```json", "").replace("```", "").strip()
            )

            start_idx = json_section.find("[")
            end_idx = json_section.rfind("]")

            if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                json_str = json_section[start_idx : end_idx + 1]
                clean_json_text = re.sub(r",\s*(]|})", r"\1", json_str)
                return json.loads(clean_json_text)

            logger.warning(
                f"Attempt {attempts + 1}: Failed to extract JSON array from Groq response payload. Retrying..."
            )
            attempts += 1

        except json.JSONDecodeError as e:
            logger.warning(
                f"Attempt {attempts + 1}: Groq returned malformed JSON ({e}). Retrying..."
            )
            attempts += 1

        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg or "503" in error_msg:
                logger.warning(
                    f"Groq Key {current_groq_idx} failed (analyze). Switching key."
                )
                current_groq_idx = (current_groq_idx + 1) % len(GROQ_API_KEYS)
                attempts += 1
            else:
                logger.error(
                    "Error occurred during code analysis with Groq.", exc_info=True
                )
                return []

    logger.error("ALL attempts to get valid JSON from Groq failed!")
    return []


async def generate_embedding(text: str) -> list[float]:
    global current_gemini_idx
    attempts = 0
    max_attempts = len(GEMINI_API_KEYS)

    while attempts < max_attempts:
        client = genai.Client(api_key=GEMINI_API_KEYS[current_gemini_idx])
        try:
            response = await client.aio.models.embed_content(
                model="gemini-embedding-2", contents=text
            )
            return response.embeddings[0].values
        except Exception as e:
            error_msg = str(e).lower()
            if (
                "429" in error_msg
                or "resource_exhausted" in error_msg
                or "503" in error_msg
            ):
                logger.warning(
                    f"Gemini Key {current_gemini_idx} failed (embedding). Switching key."
                )
                current_gemini_idx = (current_gemini_idx + 1) % len(GEMINI_API_KEYS)
                attempts += 1
            else:
                logger.error("Failed to generate embedding from Gemini.", exc_info=True)
                return []

    logger.error("ALL Gemini API Keys are exhausted!")
    return []


async def validate_user_message(text: str) -> dict:
    global current_groq_idx
    prompt = (
        "Analyze the following message. Is it a technical inquiry about code, architecture, software engineering, or the PR code review?\n"
        "If the user asks about food, weather, personal advice, or ANYTHING unrelated to programming, you MUST return is_safe: false.\n"
        f'Text: "{text}"\n\n'
        'Return ONLY raw JSON format: {"is_safe": boolean, "reason": "string"}'
    )

    attempts = 0
    max_attempts = len(GROQ_API_KEYS)

    while attempts < max_attempts:
        client = AsyncGroq(api_key=GROQ_API_KEYS[current_groq_idx])
        try:
            # noinspection PyTypeChecker
            response = await client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw_text = (response.choices[0].message.content or "").strip()

            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            return json.loads(raw_text.strip())
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg or "503" in error_msg:
                logger.warning(
                    f"Groq Key {current_groq_idx} failed (validate). Switching key."
                )
                current_groq_idx = (current_groq_idx + 1) % len(GROQ_API_KEYS)
                attempts += 1
            else:
                logger.error("User message validation failed via Groq.", exc_info=True)
                return {
                    "is_safe": False,
                    "reason": "Validation failed due to internal error",
                }

    return {"is_safe": False, "reason": "Groq API Limits Reached"}


async def generate_chat_reply(diff_hunk: str, user_text: str) -> str:
    global current_groq_idx
    prompt = (
        "You are an AI reviewer replying to a developer in a GitHub PR.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Answer ONLY the specific technical question asked by the user.\n"
        "2. DO NOT provide a general code review of the snippet unless explicitly requested.\n"
        "3. Be concise and professional. Do not use intro filler.\n\n"
        f"Context Code snippet:\n{diff_hunk}\n\nDeveloper's question: {user_text}"
    )

    attempts = 0
    max_attempts = len(GROQ_API_KEYS)

    while attempts < max_attempts:
        client = AsyncGroq(api_key=GROQ_API_KEYS[current_groq_idx])
        try:
            # noinspection PyTypeChecker
            response = await client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg or "503" in error_msg:
                logger.warning(
                    f"Groq Key {current_groq_idx} failed (chat). Switching key."
                )
                current_groq_idx = (current_groq_idx + 1) % len(GROQ_API_KEYS)
                attempts += 1
            else:
                logger.error("Failed to generate chat reply via Groq.", exc_info=True)
                return "An internal error occurred while generating the response."

    return "Service temporarily unavailable due to AI rate limits."


async def check_ai_health() -> bool:
    logger.info("Executing health check: Validating Groq & Gemini API connectivity.")
    try:
        # Check Gemini
        genai_client = genai.Client(api_key=GEMINI_API_KEYS[0])
        await genai_client.aio.models.generate_content(
            model="gemini-3.6-flash", contents="Ping."
        )

        # Check Groq
        groq_client = AsyncGroq(api_key=GROQ_API_KEYS[0])

        # noinspection PyTypeChecker
        await groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Ping."}],
            max_tokens=5,
        )
        logger.info("Health check completed: AI services connectivity is nominal.")
        return True
    except Exception:
        logger.error(
            "Health check failed: Unable to connect to AI services.", exc_info=True
        )
        return False
