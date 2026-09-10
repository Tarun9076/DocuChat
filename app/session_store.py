"""Per-session state, keyed by an opaque id stored in the user's signed session
cookie. Each browser session gets its own vector store and chat history so
concurrent users never see each other's documents or conversations.

This is in-memory by design (see README "Production notes") — it only scales
to a single worker process. A multi-worker deployment would need to move this
to a shared store (e.g. Redis) instead.
"""

import threading
import time


class SessionStore:
    def __init__(self, ttl_seconds):
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._sessions = {}

    def _purge_stale(self):
        cutoff = time.time() - self.ttl_seconds
        for sid in [sid for sid, s in self._sessions.items() if s["last_active"] < cutoff]:
            state = self._sessions.pop(sid, None)
            if state and state["db"] is not None:
                state["db"].delete_collection()

    def get(self, sid, create=False):
        with self._lock:
            self._purge_stale()
            state = self._sessions.get(sid)
            if state is None and create:
                state = {"db": None, "filename": None, "chunks": 0, "chat_history": [], "last_active": time.time()}
                self._sessions[sid] = state
            if state is not None:
                state["last_active"] = time.time()
            return state

    def set_document(self, sid, db, filename, chunk_count):
        with self._lock:
            state = self._sessions[sid]
            previous_db = state["db"]
            state["db"] = db
            state["filename"] = filename
            state["chunks"] = chunk_count
            state["chat_history"] = []
        if previous_db is not None:
            previous_db.delete_collection()

    def append_history(self, sid, human_message, ai_message):
        with self._lock:
            state = self._sessions.get(sid)
            if state is not None:
                state["chat_history"].append(human_message)
                state["chat_history"].append(ai_message)

    def clear_history(self, sid):
        with self._lock:
            state = self._sessions.get(sid)
            if state is not None:
                state["chat_history"] = []
