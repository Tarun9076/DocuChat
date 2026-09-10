import os

from dotenv import load_dotenv
from flask import Flask
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI

from app.config import Config
from app.routes import bp
from app.session_store import SessionStore

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")


def create_app(llm=None, embedding_model=None, upload_dir=None):
    """App factory. Pass `llm`/`embedding_model` overrides in tests to avoid
    hitting Azure OpenAI or downloading the real embedding model."""
    config = Config()

    if llm is None:
        config.validate_azure()
        llm = AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            azure_deployment=config.azure_deployment_name,
        )

    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    upload_dir = upload_dir or DEFAULT_UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.max_content_length
    app.config["UPLOAD_DIR"] = upload_dir
    app.config["ALLOWED_EXTENSIONS"] = config.allowed_extensions
    app.secret_key = config.secret_key

    app.extensions["llm"] = llm
    app.extensions["embedding_model"] = embedding_model
    app.extensions["session_store"] = SessionStore(config.session_ttl_seconds)

    app.register_blueprint(bp)

    return app
