# Worker

The worker package owns asynchronous CPU/GPU execution and Docker sandbox orchestration.

Current scope:

- define Celery queues;
- provide a dry-run attack task that reuses `evaluator.attacks.runner`;
- construct Docker sandbox commands for reviewed algorithm packages.

Concrete watermark algorithm execution is intentionally left for the next stage.
