from types import SimpleNamespace

import pytest
from langchain_community.embeddings import HuggingFaceEmbeddings

from app import create_app


class FakeLLM:
    """Stands in for AzureChatOpenAI in tests: no network, no Azure creds needed."""

    def __init__(self, answer="Tesla was founded in 2003."):
        self.answer = answer

    def invoke(self, messages):
        return SimpleNamespace(content=self.answer)

    def stream(self, messages):
        yield SimpleNamespace(content=self.answer)


@pytest.fixture(scope="session")
def embedding_model():
    # Real HuggingFace embeddings (small, CPU, already used by the app) — no
    # Azure calls involved, so it's safe and fast enough to share across tests.
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def app(tmp_path, embedding_model):
    return create_app(llm=FakeLLM(), embedding_model=embedding_model, upload_dir=str(tmp_path))


@pytest.fixture
def client(app):
    return app.test_client()
