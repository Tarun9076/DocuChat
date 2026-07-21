import os
import secrets
import sys
import threading
import time

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, session, stream_with_context
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {".txt", ".pdf"}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20 MB
SESSION_TTL_SECONDS = 2 * 60 * 60  # drop idle sessions' vector stores after 2 hours

os.makedirs(UPLOAD_DIR, exist_ok=True)

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if not api_key or not endpoint or not deployment_name or deployment_name == "your_azure_openai_deployment_name_here":
    print(
        "ERROR: Azure OpenAI configuration is missing or incomplete in your .env file.\n"
        "Please ensure AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_NAME are correctly set.",
        file=sys.stderr,
    )
    sys.exit(1)

model = AzureChatOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version=api_version,
    azure_deployment=deployment_name,
)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
# Must stay stable across restarts/workers so session cookies keep resolving to
# the same in-memory entry. Set FLASK_SECRET_KEY in .env for real deployments.
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)

# Per-session state, keyed by an opaque id stored in the user's signed session
# cookie. Each browser session gets its own vector store and chat history so
# concurrent users never see each other's documents or conversations.
sessions_lock = threading.Lock()
sessions = {}


def get_session_id():
    sid = session.get("sid")
    if not sid:
        sid = secrets.token_hex(16)
        session["sid"] = sid
    return sid


def purge_stale_sessions():
    cutoff = time.time() - SESSION_TTL_SECONDS
    for sid in [sid for sid, s in sessions.items() if s["last_active"] < cutoff]:
        state = sessions.pop(sid, None)
        if state and state["db"] is not None:
            state["db"].delete_collection()


def get_user_state(sid, create=False):
    with sessions_lock:
        purge_stale_sessions()
        state = sessions.get(sid)
        if state is None and create:
            state = {"db": None, "filename": None, "chunks": 0, "chat_history": [], "last_active": time.time()}
            sessions[sid] = state
        if state is not None:
            state["last_active"] = time.time()
        return state


def load_document(path, ext):
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    else:
        loader = TextLoader(path, encoding="utf-8")
    return loader.load()


@app.route("/")
def index():
    get_session_id()
    return render_template("index.html")


@app.route("/api/status")
def status():
    state = get_user_state(get_session_id())
    if state is None:
        return jsonify({"document_loaded": False, "filename": None, "chunks": 0})
    return jsonify(
        {
            "document_loaded": state["db"] is not None,
            "filename": state["filename"],
            "chunks": state["chunks"],
        }
    )


@app.route("/api/upload", methods=["POST"])
def upload():
    sid = get_session_id()

    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No file provided."}), 400

    filename = os.path.basename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '{ext}'. Only .txt and .pdf are supported."}), 400

    # Namespace the temp filename by session so concurrent uploads from
    # different users never collide on the same path on disk.
    save_path = os.path.join(UPLOAD_DIR, f"{sid}_{secrets.token_hex(4)}_{filename}")
    file.save(save_path)

    try:
        documents = load_document(save_path, ext)
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        if not chunks:
            return jsonify({"error": "No extractable text found in that document."}), 400

        # In-memory (no persist_directory): each upload gets its own
        # uniquely-named collection. langchain_chroma.Chroma defaults
        # collection_name to the fixed string "langchain" — without
        # overriding it here, every upload from every session would land
        # in that same shared collection instead of a fresh, isolated one.
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name=f"session_{sid}_{secrets.token_hex(4)}",
            collection_metadata={"hnsw:space": "cosine"},
        )

        state = get_user_state(sid, create=True)
        with sessions_lock:
            previous_db = state["db"]
            state["db"] = db
            state["filename"] = filename
            state["chunks"] = len(chunks)
            state["chat_history"] = []

        if previous_db is not None:
            previous_db.delete_collection()

        return jsonify({"filename": filename, "chunks": len(chunks)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        os.remove(save_path)


@app.route("/api/chat", methods=["POST"])
def chat():
    sid = get_session_id()
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("message") or "").strip()

    if not question:
        return jsonify({"error": "Empty message."}), 400

    state = get_user_state(sid)
    if state is None or state["db"] is None:
        return jsonify({"error": "Please upload a document first."}), 400

    db = state["db"]
    chat_history = list(state["chat_history"])

    if chat_history:
        rewrite_messages = (
            [
                SystemMessage(
                    content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."
                )
            ]
            + chat_history
            + [HumanMessage(content=f"New question: {question}")]
        )
        search_question = model.invoke(rewrite_messages).content.strip()
    else:
        search_question = question

    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(search_question)

    combined_input = f"""Based on the following documents, please answer this question: {question}

Documents:
{chr(10).join(f"- {d.page_content}" for d in docs)}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

    messages = (
        [
            SystemMessage(
                content="You are a helpful assistant that answers questions based on provided documents and conversation history."
            )
        ]
        + chat_history
        + [HumanMessage(content=combined_input)]
    )

    def generate():
        full_answer = []
        try:
            for chunk in model.stream(messages):
                token = chunk.content or ""
                if token:
                    full_answer.append(token)
                    yield token
        except Exception as exc:
            yield f"\n\n[Error: {exc}]"
            return

        answer = "".join(full_answer)
        with sessions_lock:
            state["chat_history"].append(HumanMessage(content=question))
            state["chat_history"].append(AIMessage(content=answer))

    return Response(stream_with_context(generate()), mimetype="text/plain")


@app.route("/api/reset", methods=["POST"])
def reset_chat():
    state = get_user_state(get_session_id())
    if state is not None:
        with sessions_lock:
            state["chat_history"] = []
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
