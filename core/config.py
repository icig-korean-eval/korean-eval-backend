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

    MARIADB_HOST: str = os.getenv("MARIADB_HOST")
    MARIADB_PORT: str = os.getenv("MARIADB_PORT")
    MARIADB_USERNAME: str = os.getenv("MARIADB_USERNAME")
    MARIADB_PASSWORD: str = os.getenv("MARIADB_PASSWORD")
    MARIADB_DATABASE: str = os.getenv("MARIADB_DATABASE")

    MARIADB_URL: str = (f'mariadb+pymysql://{MARIADB_USERNAME}:{MARIADB_PASSWORD}@{MARIADB_HOST}:{MARIADB_PORT}'
                        f'/{MARIADB_DATABASE}?charset=utf8mb4')

    class Config:
        case_sensitive = True


settings = Settings()