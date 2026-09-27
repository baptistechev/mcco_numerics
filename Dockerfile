# Base: Python 3.11 (TrOMA uses enum.StrEnum), CPU only
FROM python:3.11-slim

# No .pyc writes into the mounted repo (keeps the git "dirty" flag in invocations.jsonl meaningful)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# git: installs TrOMA from GitHub, and records.provenance() reads the commit of the mounted repo
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* && \
    git config --system --add safe.directory '*' && \
    pip install --no-cache-dir uv

WORKDIR /app

# Dependencies from pyproject.toml (numpy, scipy, TrOMA @ perf/vectorized-marginals, pytest).
# mcco_sim itself is not installed: it is imported from the mounted source in /app.
COPY pyproject.toml /tmp/pyproject.toml
RUN uv pip install --system --no-cache -r /tmp/pyproject.toml --extra test

# Source files mounted at runtime (outputs land in the mounted directory):
#   docker build -t mcco-sim .
#   docker run --rm -v $(pwd):/app mcco-sim
#   docker run --rm -v $(pwd):/app mcco-sim python run.py --stage all --out results --workers 16
#   docker run --rm -v $(pwd):/app mcco-sim python -m pytest
CMD ["python", "run.py", "--stage", "selftest"]
