"""
WHY Pydantic Settings?
Instead of scattered os.getenv() calls, we define a typed config class.
Benefits:
  1. Validation: app crashes immediately if a required env var is missing
  2. Type safety: IDE autocompletion works
  3. Single source of truth: import settings anywhere
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    supabase_url: str
    supabase_key: str            # anon key — used with RLS for user-scoped queries
    supabase_service_key: str    # service role key — bypasses RLS (admin operations)
    openai_api_key: str

    # Embedding config
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # LLM config
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.1  # low temp = more deterministic/factual

    # Chunking config
    chunk_size: int = 500         # tokens per chunk
    chunk_overlap: int = 50       # overlap prevents cutting mid-sentence

    class Config:
        env_file = ".env"


@lru_cache()  # singleton — only reads .env once
def get_settings() -> Settings:
    return Settings()