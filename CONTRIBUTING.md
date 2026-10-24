# Development

Use uv to manage the project environment and dependencies. Commit `uv.lock` when changing dependencies.

```sh
uv sync --locked
uv run --locked pytest
```

The tests use mocked HTTP and inference clients; a GPU, model files, live audio capture, and the `live` extra are not required. For manual live testing, install the extra and start a compatible llama.cpp server:

```sh
uv sync --locked --extra live
uv run --locked --extra live pauscribe live
```

The project selects CPU PyTorch wheels on Linux and Windows. The ASR server has its own independent GPU backend. Add runtime dependencies with `uv add`, and development dependencies with `uv add --dev`.

Build distributions with:

```sh
uv build
```

For a standalone command installation from the checkout:

```sh
uv tool install --torch-backend cpu '.[live]'
```

Model files, virtual environments, recordings, and build output must not be committed. Keep the HTTP client lightweight and import VAD dependencies only when the live command runs. GPU selection must remain configurable, without modifying the parent process environment. The service installer uses the same server argument builder as the foreground command and writes an independent user unit with absolute executable and model paths.
