# GWpy

GWpy is a Python library toolkit for gravitational-wave astrophysics.
See the `README.md` file for basic information and the `docs/` folder for
user and developer documentation.

## Tech stack

-   **Programming Language**: Python 3.10+ (development uses Python 3.12)

-   **Platform support**: Linux, macOS, Windows

-   **Versioning**: `setuptools_scm`

-   **Packaging**: `pyproject.toml` installable with `pip`/`uv` with extra dependency-groups
    for non-Python dependencies installable with `conda`.

-   **Testing**: `pytest` (config in `pyproject.toml`)

-   **Linting/formatting**: `ruff` (config in `pyproject.toml`), plus `mypy` settings

-   **Documentation**: Sphinx sources under `docs/` and `examples/`

-   **CLI entry points**: Defined in `pyproject.toml` (e.g., `gwpy-plot`)

## Repository layout (high level)

-   `gwpy/` — main package.
    Each subfolder (e.g., `gwpy/astro/`) is a subpackage with its own
    `__init__.py` and `tests/` folder.

-   `docs/` — user/developer documentation (Sphinx).

-   `examples/` — runnable examples and a lightweight integration test (`test_examples.py`).

-   `tests` are colocated inside package subfolders (e.g. `gwpy/astro/tests/`).

-   The `worktrees/` directory may be present to hold multiple git worktrees -
    ignore it during repo-wide scans and analysis.

## Coding guidelines

-   **Style: follow PEP 8 with opinionated `ruff` rules in `pyproject.toml`.
    The project runs `ruff` in CI and locally via `python -m ruff .`.
    All features should uses the latest syntax supported by the minimum Python
    version.

-   **Type checking**: all code should include comprehensive type annotations following
    PEP 563, 585, and 604.
    `mypy` is configured to check typing.

-   **Documentation**: all public classes, methods, and functions should include
    docstrings following the NumPy/SciPy docstring standard.
    Use Sphinx-compatible reStructuredText markup and follow SemBR (https://sembr.org/)
    where appropriate.
    Docstrings should include type hints even if the function is already annotated.

-   **Tests**: use `pytest`. New code must include unit tests.
    Run tests locally via:

    ```shell
    uv pip install . --group dev
    uv run pytest gwpy/ examples/
    ```

    -   Keep tests small and focused; prefer `pytest.mark.parametrize` or separate tests
        over large tests with multiple independent assertions.
        Use mocking where appropriate to avoid slow or flaky tests.

    -   Tests that require optional dependencies should be marked using `@pytest.mark.requires`.

## Commands & quick-start (recommended)

-   Preferred quick dev setup (isolated environment):

    Use `uv` to manage virtual environments (https://pypa.github.io/uv/).

    # create & activate venv
    uv venv --seed 
    . .venv/bin/activate

    # install editable package with dev extras (per `CONTRIBUTING.md`)
    uv pip install . --group dev

-   Run tests (fast subset):

    python -m pytest gwpy/ -q

-   Run ruff locally:

    python -m ruff .

-   If you need reproducible, CI-like envs, prefer conda + conda-forge binaries
    (especially when `lalsuite` or other heavy scientific packages are required).