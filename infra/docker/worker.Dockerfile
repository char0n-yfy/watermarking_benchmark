FROM python:3.11-slim

WORKDIR /app

COPY apps/worker/requirements.txt /app/apps/worker/requirements.txt
RUN pip install --no-cache-dir -r /app/apps/worker/requirements.txt

COPY apps/worker /app/apps/worker
COPY evaluator /app/evaluator

ENV PYTHONPATH=/app:/app/apps/worker
RUN mkdir -p /data/wm-bench/resources/datasets /data/wm-bench/resources/weights /data/wm-bench/runs /data/wm-bench/state
CMD ["celery", "-A", "worker_app.celery_app", "worker", "--loglevel=INFO"]
