# Development

Use uv to manage the project environment and dependencies. Commit `uv.lock` when changing dependencies.

```sh
uv sync --locked
uv run --locked pytest
```

The tests use mocked HTTP and inference clients; a GPU, model files, live audio capture, and the `live` extra are not required. For manual live testing, install the extra and start a compatible llama.cpp server:

```sh
uv sync --locked --extra live
uv run --locked --extra live wayasr live
```

For Wayland integration, use Linux and Python 3.14+ with `uv sync --locked --extra live --extra wayland`. The `wayland` extra installs asslayer from PyPI. Overlay unit tests replace the controller with a mock and never create a real surface. Run graphical verification manually in a supported Wayland session using `uv run --locked --extra live --extra wayland wayasr live --overlay`; expect bottom-centered captions and a clean shutdown on Ctrl+C.

Live VAD uses NumPy and CPU ONNX Runtime with a bundled Silero model; PyTorch is not required. The ASR server has its own independent GPU backend. Add runtime dependencies with `uv add`, and development dependencies with `uv add --dev`.

Build distributions with:

```sh
uv build
```

For a standalone command installation from the checkout:

```sh
uv tool install '.[live]'
```

## Publishing to PyPI

CI runs the tests on Python 3.10, 3.12, and 3.14. Publishing runs separately when a GitHub release is published. The release tag must match the package version, for example `v0.1.0` for version `0.1.0` in `pyproject.toml`.

Configure a PyPI Trusted Publisher with these values:

| Field | Value |
| --- | --- |
| PyPI project name | `wayasr` |
| Owner | `sitiyou` |
| Repository name | `wayasr` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

The GitHub repository must have an environment named `pypi`. Use a pending publisher for the first release, or add the publisher in the existing PyPI project's publishing settings for later setup. No PyPI API token or repository secret is needed.

Before releasing, update the version in `pyproject.toml`, refresh `uv.lock` with `uv lock`, and commit the changes. Ensure the commits are available in the GitHub repository and CI passes, then publish a GitHub release using the matching version tag. Creating a tag alone or saving a draft release does not trigger publishing. Published PyPI versions cannot be overwritten; each subsequent release needs a new version.

The publishing workflow verifies the version, runs tests, builds and checks both distributions, and smoke-tests the wheel. A separate job publishes the resulting artifacts using OIDC with the `pypi` environment. Only that job receives `id-token: write`.

Model files, virtual environments, recordings, and build output must not be committed. Keep the HTTP client lightweight and import VAD dependencies only when the live command runs. GPU selection must remain configurable, without modifying the parent process environment. The service installer uses the same server argument builder as the foreground command and writes an independent user unit with absolute executable and model paths.
