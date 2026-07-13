# Contributing

Evidence Bench welcomes focused fixes, new deterministic validators, scientific
benchmark cases, task profiles, and A2A interoperability improvements.

## Development

```bash
uv sync --extra dev --extra analysis
npm ci --ignore-scripts
uv run pytest -m "not live"
```

Live model/MCP tests are opt-in and must never be added to the default CI path.
New tool capabilities require a threat-model update, a typed interface, policy
tests, and a sandbox test. New scientific validators should include both a known
positive and a known negative fixture.

Please keep pull requests narrow. Do not include credentials, private endpoints,
patient data, or generated run directories. By contributing, you agree that your
contribution is licensed under Apache-2.0.
