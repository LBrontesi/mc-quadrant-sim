FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md uv.lock ./
COPY src ./src
COPY web ./web
COPY web_app.py ./

RUN python -m pip install --no-cache-dir uv \
    && uv sync --frozen --extra data

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860

CMD ["python", "web_app.py"]
