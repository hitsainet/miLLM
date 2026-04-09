# Security Policy

## Supported Versions

miLLM is under active development. Security fixes are applied to the latest version on the `main` branch.

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub issues.

Report vulnerabilities by opening a [GitHub Security Advisory](https://github.com/Onegaishimas/miLLM/security/advisories/new) on this repository. This keeps the report private until a fix is available.

Include:
- A description of the vulnerability and its potential impact
- Steps to reproduce
- Any suggested remediation if you have one

You can expect an acknowledgement within 7 days and a status update within 30 days.

## Deployment Security Notes

miLLM is designed for use on a trusted local network or private research cluster. Before deploying:

- **Change all default credentials** in the K8s manifest or Docker Compose file before deploying. The default values (`changeme`) are placeholders only.
- **Use Kubernetes Secrets** or environment-variable injection (not plaintext manifest values) for credentials in production deployments.
- **Restrict network access** — the API does not require authentication by default and is not designed to be exposed to the public internet.
- **CORS origins** — set `CORS_ORIGINS` to only the domains that need access to the API.
- **The `/v1/chat/completions` endpoint** has no authentication. Anyone with network access can send inference requests, including steering configurations. Deploy behind a network boundary or add authentication at the reverse proxy layer if broader access is needed.
