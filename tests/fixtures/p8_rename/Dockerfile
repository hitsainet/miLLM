FROM python:3.13-slim
COPY oldpkg/__init__.py oldpkg/__init__.py
COPY oldpkg/ /app/oldpkg/
RUN groupadd --gid 1000 oldpkg && useradd --uid 1000 --gid 1000 oldpkg
RUN python -m oldpkg.rules.runner_worker fetch /app/bundle
USER oldpkg
CMD ["python", "-m", "uvicorn", "oldpkg.main:app"]
