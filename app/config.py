from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env into the environment, so Pydantic AI also finds GROQ_API_KEY.
load_dotenv()


class Settings(BaseSettings):
    gmail_address: str
    gmail_app_password: str
    database_url: str
    classifier_model: str = "groq:openai/gpt-oss-20b"
    agent_model: str = "groq:openai/gpt-oss-120b"
    poll_interval_seconds: int = 60
    dashboard_username: str = "admin"
    dashboard_password: str = ""


settings = Settings()