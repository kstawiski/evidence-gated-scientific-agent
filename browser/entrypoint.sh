#!/usr/bin/env bash
set -Eeuo pipefail

profile=/browser-data/profile
downloads=/browser-data/downloads
home=/browser-data/home
downloads_mode=${BROWSER_DOWNLOADS_MODE:-2750}

if [[ $(id -u) == 0 ]]; then
  mkdir -p "$profile" "$downloads" "$home" "$home/.cache" "$home/.config" /tmp/.X11-unix
  chown -R browser:browser /browser-data
  chown -R browser:browser-downloads "$downloads"
  chmod 1777 /tmp/.X11-unix
  chmod 0700 /browser-data "$profile" "$home"
  chmod "$downloads_mode" "$downloads"
  exec gosu browser:browser "$0" "$@"
fi

mkdir -p "$profile" "$downloads" "$home"
chmod 0700 "$profile" "$home"
chmod "$downloads_mode" "$downloads"

# Chromium leaves these locks behind after a hard container stop. They are safe
# to remove because Compose runs exactly one browser against this profile.
rm -f "$profile"/SingletonCookie "$profile"/SingletonLock "$profile"/SingletonSocket

children=()
# Invoked indirectly by the signal/exit trap below.
# shellcheck disable=SC2329
cleanup() {
  trap - TERM INT EXIT
  if ((${#children[@]})); then
    kill -TERM "${children[@]}" 2>/dev/null || true
    wait "${children[@]}" 2>/dev/null || true
  fi
}
trap cleanup TERM INT EXIT

Xvfb "$DISPLAY" -screen 0 "${BROWSER_GEOMETRY:-1600x1000x24}" -ac +extension GLX +render -noreset &
children+=("$!")
for _ in {1..100}; do
  [[ -S "/tmp/.X11-unix/X${DISPLAY#:}" ]] && break
  sleep 0.05
done
[[ -S "/tmp/.X11-unix/X${DISPLAY#:}" ]] || { echo "Xvfb did not start" >&2; exit 1; }

openbox-session &
children+=("$!")

# VNC never leaves the container. noVNC is the sole interactive ingress and is
# intentionally passwordless for trusted private-network deployments.
x11vnc -display "$DISPLAY" -rfbport 5900 -localhost -forever -shared -nopw -quiet &
children+=("$!")
websockify --web=/usr/share/novnc/ 6080 127.0.0.1:5900 &
children+=("$!")

# Current Chromium intentionally binds DevTools to loopback even when an
# alternate address is requested. This container-local reverse proxy makes CDP
# visible on the isolated Compose network, rewrites Chromium's loopback
# WebSocket URL, and is never published on the host.
nginx -c /etc/nginx/nginx.conf -g 'daemon off;' &
children+=("$!")

chromium \
  --user-data-dir="$profile" \
  --proxy-server=http://browser-egress:3128 \
  --proxy-bypass-list='<-loopback>' \
  --remote-debugging-address=0.0.0.0 \
  --remote-debugging-port=9223 \
  --remote-allow-origins='*' \
  --no-sandbox \
  --no-first-run \
  --no-default-browser-check \
  --password-store=basic \
  --disable-breakpad \
  --disable-dev-shm-usage \
  --disable-features=Translate \
  --start-maximized \
  about:blank &
children+=("$!")

wait -n "${children[@]}"
status=$?
echo "Managed browser component exited with status $status" >&2
exit "$status"
