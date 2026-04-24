# syntax=docker/dockerfile:1.7

# ---- Stage 1: builder ---------------------------------------------------- #
# Compiles wheels once so the final image stays small and cache-friendly.
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build deps for scientific wheels (xgboost, scipy, numpy on some archs).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy metadata first so the dependency layer is cached when only source changes.
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip wheel \
    && pip wheel --wheel-dir /wheels .


# ---- Stage 2: runtime ---------------------------------------------------- #
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Bind to all interfaces by default inside the container; compose/k8s
    # publish the port. Override via MLAB_API__HOST if you need otherwise.
    MLAB_API__HOST=0.0.0.0 \
    MLAB_API__PORT=8000 \
    MLAB_DASHBOARD__PORT=8501 \
    MLAB_DATABASE__PATH=/data/platform.db \
    MLAB_MODELS__ARTIFACTS_DIR=/data/artifacts \
    MLAB_MODELS__MODEL_A_PATH=/data/artifacts/model_a.joblib \
    MLAB_MODELS__MODEL_B_PATH=/data/artifacts/model_b.joblib \
    MLAB_MODELS__BASELINE_METRICS_PATH=/data/artifacts/baseline_metrics.json

# libgomp1 is an XGBoost runtime requirement. curl is used by the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system mlab \
    && useradd --system --gid mlab --home /home/mlab --create-home mlab

# Install the package from pre-built wheels.
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels ml-ab-platform \
    && rm -rf /wheels

# Copy config files the app reads at runtime.
WORKDIR /app
COPY --chown=mlab:mlab configs ./configs

# Persistent volume for SQLite + trained model artifacts. Containers are
# ephemeral; models and the experiment DB must outlive them.
RUN mkdir -p /data/artifacts && chown -R mlab:mlab /data /app
VOLUME ["/data"]

USER mlab

EXPOSE 8000 8501

# Default: run the FastAPI gateway. Override with `docker run ... mlab dashboard`
# or the docker-compose services.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:${MLAB_API__PORT}/health || exit 1

ENTRYPOINT ["mlab"]
CMD ["serve"]
