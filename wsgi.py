"""Production WSGI entrypoint, e.g.:

    gunicorn -w 1 --threads 4 -b 0.0.0.0:5000 wsgi:app        # Linux/mac
    waitress-serve --host=0.0.0.0 --port=5000 wsgi:app        # Windows

1 worker is intentional: session state (uploaded doc + chat history) lives in
memory, keyed by session cookie, so it doesn't survive being split across
multiple processes. Threads still give you concurrency within that worker.
See README "Production notes" for details.
"""

from app import create_app

app = create_app()
