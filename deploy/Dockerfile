# One image for scorer / simulator / trainer; the entrypoint picks the role.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH" \
    MLFLOW_DISABLE_AGENT_HINT=1
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra ml --extra serve --no-install-project
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
RUN uv sync --frozen --no-dev --extra ml --extra serve
COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["scorer"]
