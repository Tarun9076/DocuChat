import os
import secrets

from flask import Blueprint, Response, current_app, jsonify, render_template, request, session, stream_with_context
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.rag import build_vectorstore, load_and_split

bp = Blueprint("main", __name__)


def get_session_id():
    sid = session.get("sid")
    if not sid:
        sid = secrets.token_hex(16)
        session["sid"] = sid
    return sid


@bp.route("/")
def index():
    get_session_id()
    return render_template("index.html")


@bp.route("/api/status")
def status():
    store = current_app.extensions["session_store"]
    state = store.get(get_session_id())
    if state is None:
        return jsonify({"document_loaded": False, "filename": None, "chunks": 0})
    return jsonify(
        {
            "document_loaded": state["db"] is not None,
            "filename": state["filename"],
            "chunks": state["chunks"],
        }
    )


@bp.route("/api/upload", methods=["POST"])
def upload():
    store = current_app.extensions["session_store"]
    embedding_model = current_app.extensions["embedding_model"]
    upload_dir = current_app.config["UPLOAD_DIR"]
    allowed_extensions = current_app.config["ALLOWED_EXTENSIONS"]

    sid = get_session_id()

    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No file provided."}), 400

    filename = os.path.basename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported file type '{ext}'. Only .txt and .pdf are supported."}), 400

    # Namespace the temp filename by session so concurrent uploads from
    # different users never collide on the same path on disk.
    save_path = os.path.join(upload_dir, f"{sid}_{secrets.token_hex(4)}_{filename}")
    file.save(save_path)

    try:
        chunks = load_and_split(save_path, ext)

        if not chunks:
            return jsonify({"error": "No extractable text found in that document."}), 400

        db = build_vectorstore(chunks, embedding_model, sid)

        store.get(sid, create=True)
        store.set_document(sid, db, filename, len(chunks))

        return jsonify({"filename": filename, "chunks": len(chunks)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        os.remove(save_path)


@bp.route("/api/chat", methods=["POST"])
def chat():
    store = current_app.extensions["session_store"]
    llm = current_app.extensions["llm"]

    sid = get_session_id()
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("message") or "").strip()

    if not question:
        return jsonify({"error": "Empty message."}), 400

    state = store.get(sid)
    if state is None or state["db"] is None:
        return jsonify({"error": "Please upload a document first."}), 400

    db = state["db"]
    chat_history = list(state["chat_history"])

    if chat_history:
        rewrite_messages = (
            [
                SystemMessage(
                    content=(
                        "Given the chat history, rewrite the new question to be standalone "
                        "and searchable. Just return the rewritten question."
                    )
                )
            ]
            + chat_history
            + [HumanMessage(content=f"New question: {question}")]
        )
        search_question = llm.invoke(rewrite_messages).content.strip()
    else:
        search_question = question

    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(search_question)

    combined_input = f"""Based on the following documents, please answer this question: {question}

Documents:
{chr(10).join(f"- {d.page_content}" for d in docs)}

Please provide a clear, helpful answer using only the information from these documents. If you \
can't find the answer in the documents, say "I don't have enough information to answer that \
question based on the provided documents."
"""

    messages = (
        [
            SystemMessage(
                content=(
                    "You are a helpful assistant that answers questions based on "
                    "provided documents and conversation history."
                )
            )
        ]
        + chat_history
        + [HumanMessage(content=combined_input)]
    )

    def generate():
        full_answer = []
        try:
            for chunk in llm.stream(messages):
                token = chunk.content or ""
                if token:
                    full_answer.append(token)
                    yield token
        except Exception as exc:
            yield f"\n\n[Error: {exc}]"
            return

        answer = "".join(full_answer)
        store.append_history(sid, HumanMessage(content=question), AIMessage(content=answer))

    return Response(stream_with_context(generate()), mimetype="text/plain")


@bp.route("/api/reset", methods=["POST"])
def reset_chat():
    store = current_app.extensions["session_store"]
    store.clear_history(get_session_id())
    return jsonify({"ok": True})
