FROM python:3.12-slim AS native-builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY src/mc_quadrants/native_simulation.cpp ./native_simulation.cpp
RUN g++ -O3 -std=c++17 -fPIC -shared native_simulation.cpp -o _native_sim.so

FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user pyproject.toml README.md uv.lock ./
COPY --chown=user src ./src
COPY --from=native-builder --chown=user /build/_native_sim.so ./src/mc_quadrants/_native_sim.so
COPY --chown=user web ./web
COPY --chown=user web_app.py ./

RUN python -m pip install --no-cache-dir --user uv \
    && PATH="$HOME/.local/bin:$PATH" uv sync --frozen --extra data

ENV PATH="$HOME/app/.venv/bin:$HOME/.local/bin:$PATH"

EXPOSE 7860

CMD ["python", "web_app.py"]
