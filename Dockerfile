FROM python:3.11-slim

WORKDIR /srv/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY wsgi.py .

EXPOSE 5000

# 1 worker is deliberate: session state (uploaded doc + chat history) lives in
# memory keyed by session cookie, so it can't be split across processes.
# Threads give concurrency within that worker. See README "Production notes".
CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:5000", "wsgi:app"]
