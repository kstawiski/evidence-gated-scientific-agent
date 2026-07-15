# syntax=docker/dockerfile:1.7

FROM debian:bookworm-slim

ARG BROWSER_UID=10002
ARG BROWSER_GID=10002
ARG DOWNLOAD_GID=10001

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:99 \
    HOME=/browser-data/home \
    XDG_CACHE_HOME=/browser-data/home/.cache \
    XDG_CONFIG_HOME=/browser-data/home/.config \
    BROWSER_GEOMETRY=1600x1000x24

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        chromium \
        curl \
        fonts-dejavu-core \
        fonts-liberation \
        fonts-noto-color-emoji \
        fonts-noto-core \
        gosu \
        nginx-light \
        novnc \
        openbox \
        socat \
        squid \
        tini \
        websockify \
        x11vnc \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "$BROWSER_GID" browser \
    && groupadd --gid "$DOWNLOAD_GID" browser-downloads \
    && useradd --uid "$BROWSER_UID" --gid "$BROWSER_GID" --home-dir /browser-data/home --no-create-home --shell /usr/sbin/nologin browser \
    && usermod --append --groups browser-downloads browser \
    && install -d -o browser -g browser -m 0700 \
        /browser-data \
        /browser-data/downloads \
        /browser-data/home \
        /browser-data/profile \
    && install -d -m 0755 /etc/chromium/policies/managed

COPY browser/policies.json /etc/chromium/policies/managed/evidence-bench.json
COPY browser/nginx.conf /etc/nginx/nginx.conf
COPY browser/squid.conf /etc/squid/evidence-bench.conf
COPY --chmod=0755 browser/entrypoint.sh /usr/local/bin/evidence-browser

WORKDIR /browser-data

EXPOSE 6080 9222
VOLUME ["/browser-data"]

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD ["/bin/sh", "-c", "curl -fsS http://127.0.0.1:9222/json/version >/dev/null && curl -fsS http://127.0.0.1:6080/vnc.html >/dev/null"]

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/evidence-browser"]
