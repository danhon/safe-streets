# syntax=docker/dockerfile:1.7

# The dependencies and the detector, in layers that change only with
# uv.lock or the pinned model; the code (src/) is copied last.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS deps
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN touch README.md
ENV UV_PROJECT_ENVIRONMENT=/app/.venv UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# The default detector, RTMDet-tiny as an OpenVINO IR: from the build
# context's models/rtmdet-tiny/ when it's there (scripts/export_rtmdet.py),
# else downloaded from DETECTOR_URL (by default this repository's release
# asset). Either way it must match the sha256
# pinned in detector.py, or the build stops. A cold start needs no network.
ARG DETECTOR_URL="https://github.com/danhon/safe-streets/releases/download/detector-v1/rtmdet-tiny.tar.gz"
COPY scripts/fetch_detector.py /build/scripts/
COPY src/safe_streets/detector.py /build/src/safe_streets/
COPY model[s] /build/models/
RUN python3 /build/scripts/fetch_detector.py --from /build/models --url "${DETECTOR_URL}" /models \
 && rm -rf /build

# The project itself, editable: a .pth naming /app/src and the script, which
# only imports safe_streets.main. A stub package is enough here.
RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p src/safe_streets && touch src/safe_streets/__init__.py \
 && uv sync --frozen --no-dev \
 && rm -rf src


FROM python:3.12-slim-bookworm
# ffmpeg/ffprobe: the folder source cuts clips from segments with them.
# INTEL_GPU=1 (compose.gpu.yml) adds the compute runtime OpenVINO's GPU
# plugin needs; without it the detector runs on the CPU.
ARG INTEL_GPU=0
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 \
      $([ "${INTEL_GPU}" = "1" ] && echo intel-opencl-icd) \
 && rm -rf /var/lib/apt/lists/*
ARG APP_UID=10001
RUN useradd -m -u ${APP_UID} app
WORKDIR /app
COPY --from=deps --chown=app:app /app/.venv /app/.venv
COPY --from=deps --chown=app:app /models /models
ENV PATH="/app/.venv/bin:${PATH}" \
    DATA_DIR="/data" \
    MODELS_DIR="/models" \
    OV_CACHE_DIR="/data/ov-cache" \
    WEB_HOST="0.0.0.0" \
    WEB_PORT="8080"
RUN mkdir -p /data && chown -R app:app /data
COPY --chown=app:app pyproject.toml /app/pyproject.toml
COPY --chown=app:app src/ /app/src/
EXPOSE 8080
USER app
ARG GIT_SHA=dev
ENV BUILD_SHA="${GIT_SHA}"
# nice 10: the analysis is batch work; whatever else the machine runs comes first.
CMD ["nice", "-n", "10", "safe-streets"]
