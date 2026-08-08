FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user pyproject.toml README.md uv.lock ./
COPY --chown=user src ./src
COPY --chown=user gradio_app.py streamlit_app.py ./

RUN python -m pip install --no-cache-dir --user uv \
    && PATH="$HOME/.local/bin:$PATH" uv sync --frozen --extra data

ENV PATH="$HOME/app/.venv/bin:$HOME/.local/bin:$PATH"

EXPOSE 7860
EXPOSE 8501

CMD ["python", "gradio_app.py"]
