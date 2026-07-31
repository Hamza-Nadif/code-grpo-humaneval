FROM python:3.11-slim

RUN useradd --create-home --uid 10001 runner
USER runner
WORKDIR /workspace
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-I", "/workspace/candidate.py"]
