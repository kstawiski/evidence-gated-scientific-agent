# Security policy

## Reporting a vulnerability

Do not open a public issue for a suspected sandbox escape, authentication bypass,
credential leak, or unsafe tool-policy behavior. Use GitHub's private security
advisory flow for the repository. Maintainers should acknowledge a report within
seven days and coordinate disclosure after a fix is available.

## Deployment boundary

Evidence Bench is designed for trusted single-organization deployments. It uses
HTTP Basic authentication for the browser by default and a separate bearer token
for A2A; place it behind TLS before exposing it beyond loopback or a private
network. Setting `WEB_AUTH_ENABLED=false` intentionally opens the UI and REST API
to every client that can reach the bound address. Use that mode only on a trusted,
access-controlled LAN or Tailnet; it is not an authentication mechanism.
Python/R execution is delegated to the non-published sandbox-worker container and
nested inside bubblewrap. Package installation uses a separate non-published
worker with outbound network access but no research-data or secret mounts. Both
workers intentionally receive namespace setup capabilities and must remain on
their dedicated Docker networks. Container isolation is still part of the
security boundary. Never mount the Docker socket, host root, SSH keys, or
credential directories into any service.

Canonical-registry validation constrains the requested PyPI, CRAN, and
Bioconductor packages. It does not prove that registry packages or their build
hooks are benign, and build hooks can make secondary network requests. Installed
code is subsequently mounted read-only into an offline analysis sandbox. Use an
egress proxy and an internal package mirror when stronger supply-chain policy is
required.
