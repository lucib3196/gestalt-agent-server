from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    model: str
    embedding_model: str


@lru_cache
def get_settings() -> Settings:
    if not Settings().model:  # type: ignore
        raise ValueError("Failed to load AI model. Must be set in ENV")
    return Settings()  # type: ignore


if __name__ == "__main__":
    print(get_settings())
