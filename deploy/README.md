# Deployment overlays

`compose.yaml` is the portable release surface. Site-specific routes and credentials
stay outside the repository in a mode-0600 `.env` file.

When a model endpoint is reachable only through a bastion, copy
`evidence-bench-qwen-tunnel.service.example` into the deploying user's systemd
directory and replace the uppercase placeholders. Bind the local forward to the
Docker bridge address rather than `0.0.0.0`; containers can then use
`http://host.docker.internal:<LOCAL_PORT>/v1` without exposing the model tunnel on
the LAN or Tailnet.

```bash
systemctl --user daemon-reload
systemctl --user enable --now evidence-bench-qwen-tunnel.service
```

For a boot-persistent lab installation, copy
`evidence-bench-compose.service.example` into `/etc/systemd/system`, replace
`INSTALL_DIRECTORY` with the absolute checkout path, then enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now evidence-bench.service
```

The unit complements Compose's `restart: unless-stopped`: systemd reconciles the
whole project after boot, while Docker restarts individual failed containers.
Keep the owner-only `.env` alongside the checkout and never copy it into an image.

Do not mount the Docker socket or a host home directory into either service.

For durable bind mounts, set these values in the private `.env`:

```text
EVIDENCE_BENCH_DATA_PATH=/durable/path/evidence-bench/data
EVIDENCE_BENCH_ENVIRONMENTS_PATH=/durable/path/evidence-bench/environments
```

The web UID (10001) must be able to write the data path. The package worker owns
the environment path through its confined root controller. On the UMED `s1`
installation both paths live below `/data-onix/EvidenceBench`; keep application
source and Docker build cache on local storage because Onix is a shared,
capacity-constrained NFS mount.

`provision_env.py` creates or refreshes an owner-only Compose `.env` without
printing credentials. It preserves previously generated browser/A2A/worker secrets,
accepts endpoint settings through its process environment, and copies only the
allow-listed Context7 and Brave values from an optional owner-managed MCP env file.
