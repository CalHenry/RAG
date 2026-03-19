# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ── Install uv ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
# Download the latest installer
ADD https://astral.sh/uv/0.10.11/install.sh /uv-installer.sh
# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install dependencies (cached layer) ───────────────────────────────────────
# Ensure uv copies files instead of creating symlinks
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY uv.lock pyproject.toml ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project


# ── Copy source code ──────────────────────────────────────────────────────────
COPY src/ ./src/
COPY scripts/ ./scripts/

# ── Create expected directories ───────────────────────────────────────────────
# These are the mount points for volumes. Creating them ensures the paths
# exist even if the user forgets to mount something.
RUN mkdir -p /app/models /app/src/rag/data

# ── Environment variables ─────────────────────────────────────────────────────
# PYTHONUNBUFFERED: print() output appears immediately in docker logs (no buffer).
# PYTHONDONTWRITEBYTECODE: don't create .pyc files inside the container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Default command ───────────────────────────────────────────────────────────
CMD ["uv", "run", "python", "scripts/run_query.py", "--help"]
