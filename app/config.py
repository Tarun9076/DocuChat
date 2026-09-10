import os
import secrets


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or incomplete."""


class Config:
    """Env-driven app configuration, validated once at startup."""

    def __init__(self):
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
        self.azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        # Must stay stable across restarts/workers so session cookies keep resolving to
        # the same in-memory entry. Set FLASK_SECRET_KEY in .env for real deployments.
        self.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)

        self.max_content_length = 20 * 1024 * 1024  # 20 MB
        self.session_ttl_seconds = 2 * 60 * 60  # drop idle sessions' vector stores after 2 hours
        self.allowed_extensions = {".txt", ".pdf"}

    def validate_azure(self):
        if (
            not self.azure_api_key
            or not self.azure_endpoint
            or not self.azure_deployment_name
            or self.azure_deployment_name == "your_azure_openai_deployment_name_here"
        ):
            raise ConfigError(
                "Azure OpenAI configuration is missing or incomplete. Please ensure "
                "AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_NAME "
                "are correctly set in your .env file."
            )
