FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md uv.lock ./
COPY src ./src
COPY gradio_app.py streamlit_app.py ./

RUN python -m pip install --no-cache-dir uv \
    && uv sync --frozen --all-extras

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860

CMD ["python", "gradio_app.py"]
