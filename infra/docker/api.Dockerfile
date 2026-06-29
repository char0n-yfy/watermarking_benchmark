FROM python:3.11-slim

WORKDIR /app

COPY apps/api/requirements.txt /app/apps/api/requirements.txt
RUN pip install --no-cache-dir -r /app/apps/api/requirements.txt

COPY apps/api /app/apps/api
COPY evaluator /app/evaluator

ENV PYTHONPATH=/app:/app/apps/api
RUN mkdir -p /data/wm-bench/resources/datasets /data/wm-bench/resources/weights /data/wm-bench/runs /data/wm-bench/state
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/apps/api"]
