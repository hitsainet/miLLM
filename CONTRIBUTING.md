# Contributing to miLLM

Thank you for your interest in contributing to miLLM. This project is a self-hosted LLM inference server built for mechanistic interpretability research and contributions are welcome.

## Ways to Contribute

- **Bug reports** — Open an issue describing the problem, your environment (GPU, VRAM, OS, model), and steps to reproduce.
- **Feature requests** — Open an issue describing the use case and how it fits the interpretability research workflow.
- **Code contributions** — See the development setup below.
- **Documentation** — Corrections and additions to the [documentation site](https://docs-millm.hitsai.net/) are welcome.

## Development Setup

### Prerequisites

- Python 3.13 (the image's version; `requires-python` and CI agree, `tests/unit/test_python_version_is_one_version.py`)
- Node.js 22 (LTS)
- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit

### Running Locally

```bash
# Clone the repository
git clone https://github.com/hitsainet/miLLM.git
cd miLLM

# Start infrastructure services
docker compose up -d postgres redis

# Start the backend (a 3.13 venv: `requires-python` refuses an older interpreter)
python3.13 -m venv venv && . venv/bin/activate
# CPU torch first (the lock's version), then everything else from the lock, as CI does
pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
pip install -c constraints.txt -e ".[dev]"
python docker/check_installed_against_constraints.py   # the venv matches constraints.txt
uvicorn millm.main:app --reload --port 8000

# Start the admin UI
cd admin-ui
npm ci
npm run dev  # http://localhost:3000
```

### Running Tests

```bash
# Unit tests (no external dependencies required)
pytest tests/ -m unit -v

# Integration tests (requires running PostgreSQL)
DATABASE_URL=postgresql+asyncpg://millm:...@localhost:5432/millm \
  pytest tests/ -m integration -v
```

## Code Standards

**Python:**
- Formatter: Ruff format
- Linter: Ruff
- Type checker: MyPy (strict)

**TypeScript:**
- Formatter: Prettier
- Linter: ESLint
- All components must be strictly typed

Run linters before submitting:

```bash
# Backend
ruff check millm/
ruff format --check millm/
mypy millm/

# Admin UI
cd admin-ui
npm run lint
npm run typecheck
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes with tests where applicable.
3. Ensure all existing tests pass.
4. Submit a pull request with a clear description of what changed and why.
5. Link any related issues in the PR description.

Maintainers review PRs on a best-effort basis. Please be patient.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add probe monitoring history export
fix: correct SAE hook cleanup on model unload
docs: add steering configuration examples
refactor: extract forward hook logic into dedicated module
test: add unit tests for SAE service attachment
```

## Questions

Open a [GitHub Discussion](https://github.com/hitsainet/miLLM/discussions) for questions about the codebase, research questions, or general discussion about SAE-based steering and monitoring.
