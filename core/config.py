from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"

    MEDIA_DIR: str = os.path.join(os.path.dirname(__file__), "../media")
    LOG_DIR: str = os.path.join(os.path.dirname(__file__), "../log")
    if not os.path.exists(MEDIA_DIR):
        os.makedirs(MEDIA_DIR, exist_ok=True)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)

    AUTH_KEY: str = os.getenv('AUTH_KEY')
    HUGGINGFACE_KEY: str = os.getenv('HUGGINGFACE_KEY')
    OLLAMA_KEY: str = os.getenv('OLLAMA_KEY')

    DB_PATH: str = './media/chat.db'

    class Config:
        case_sensitive = True


settings = Settings()