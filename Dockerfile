# syntax=docker/dockerfile:1.7

FROM node:22.18.0-bookworm-slim AS node-build
WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts

FROM ghcr.io/astral-sh/uv:0.10.2 AS uv

FROM python:3.12.11-slim-bookworm AS runtime-deps

ARG APP_UID=10001
ARG APP_GID=10001

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    SCIENTIFIC_AGENT_DATA_DIR=/data \
    SCIENTIFIC_AGENT_PYTHON=/opt/venv/bin/python3 \
    SCIENTIFIC_AGENT_PYTHON_PREFIX=/usr/local \
    SCIENTIFIC_AGENT_PYTHON_PACKAGES=/opt/venv/lib/python3.12/site-packages \
    SCIENTIFIC_AGENT_RSCRIPT=/usr/bin/Rscript \
    SCIENTIFIC_AGENT_R_LIBRARY=/usr/local/lib/R/site-library \
    HOME=/tmp/home \
    WEB_HOST=0.0.0.0 \
    WEB_PORT=8080

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bubblewrap \
        ca-certificates \
        libcurl4 \
        libfontconfig1 \
        libfreetype6 \
        libfribidi0 \
        libgdal32 \
        libgeos-c1v5 \
        libgit2-1.5 \
        libglpk40 \
        libgsl27 \
        libharfbuzz0b \
        libhdf5-103-1 \
        libicu72 \
        libjpeg62-turbo \
        libnetcdf19 \
        libpng16-16 \
        libproj25 \
        libssl3 \
        libtiff6 \
        libudunits2-0 \
        libxml2 \
        passwd \
        poppler-utils \
        r-base \
        r-cran-data.table \
        r-cran-dplyr \
        r-cran-ggplot2 \
        r-cran-jsonlite \
        r-cran-survival \
        tesseract-ocr \
        tini \
        util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && /usr/sbin/groupadd --gid "$APP_GID" evidence \
    && /usr/sbin/useradd --uid "$APP_UID" --gid "$APP_GID" --no-create-home --shell /usr/sbin/nologin evidence \
    && install -d -o evidence -g evidence -m 0700 /data /tmp/home

COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=node-build /usr/local/bin/node /usr/local/bin/node

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra analysis --no-install-project
COPY --from=node-build /build/node_modules ./node_modules

FROM runtime-deps AS app-base

COPY README.md ./
COPY scientific_agent ./scientific_agent
COPY integrations/a2a ./integrations/a2a
COPY skills/evidence-bench ./skills/evidence-bench
RUN uv sync --frozen --no-dev --extra analysis \
    && rm -rf /root/.cache/uv

USER evidence:evidence
EXPOSE 8080 8090
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3)"]

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["scientific-agent-web"]

FROM runtime-deps AS package-deps

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libbz2-dev \
        libcairo2-dev \
        libcurl4-openssl-dev \
        libfontconfig1-dev \
        libfreetype6-dev \
        libfribidi-dev \
        libgdal-dev \
        libgeos-dev \
        libgit2-dev \
        libglpk-dev \
        libgsl-dev \
        libharfbuzz-dev \
        libhdf5-dev \
        libicu-dev \
        libjpeg-dev \
        libpng-dev \
        libnetcdf-dev \
        libpcre2-dev \
        libproj-dev \
        libreadline-dev \
        libssl-dev \
        libtiff-dev \
        libxml2-dev \
        libudunits2-dev \
        libxt-dev \
        libzstd-dev \
        liblzma-dev \
        pkg-config \
        r-base-dev \
    && rm -rf /var/lib/apt/lists/*

FROM package-deps AS environment-worker

COPY README.md ./
COPY scientific_agent ./scientific_agent
COPY integrations/a2a ./integrations/a2a
COPY skills/evidence-bench ./skills/evidence-bench
RUN uv sync --frozen --no-dev --extra analysis \
    && rm -rf /root/.cache/uv

USER root
EXPOSE 8091
VOLUME ["/environments"]
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["scientific-agent-environment-worker"]

FROM app-base AS runtime
