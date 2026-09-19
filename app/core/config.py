from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_keys: str
    groq_api_keys: str
    github_app_id: str
    github_private_key_path: str = "github-key.pem"
    database_url: str

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def gemini_keys_list(self) -> list[str]:
        return [
            k.strip().strip(" \"'")
            for k in self.gemini_api_keys.split(",")
            if k.strip()
        ]

    @property
    def groq_keys_list(self) -> list[str]:
        return [
            k.strip().strip(" \"'") for k in self.groq_api_keys.split(",") if k.strip()
        ]


settings = Settings()
