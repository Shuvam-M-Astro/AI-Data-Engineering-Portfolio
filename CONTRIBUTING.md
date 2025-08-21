# Contributing

Thank you for considering contributing to this AI Data Engineering Portfolio!

## Getting Started

1. Fork the repository and create your feature branch from `main`.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or: source .venv/bin/activate  # macOS/Linux
   pip install -r requirements-common.txt
   ```
3. Install dev tools and pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Coding Standards

- Use Black for formatting; isort for imports; Ruff for linting.
- Keep functions short and purposeful with clear names.
- Prefer configuration-driven experiments.
- Add docstrings to modules, classes, and functions that are non-trivial.

## Making Changes

- Add or update a project under the appropriate domain directory.
- Provide a `README.md` and `requirements.txt` inside the project directory.
- If you add a runnable experiment, consider wiring it into `runner/run.py`.
- Include small synthetic datasets or links to external data sources, not large files.

## Testing and CI

- Ensure `ruff`, `black --check`, and `isort --check-only` pass locally.
- CI will verify code style on pushes and pull requests.

## Pull Requests

- Use a clear title and description of the change and motivation.
- Reference relevant issues if applicable.
- Keep PRs focused and reasonably small.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
