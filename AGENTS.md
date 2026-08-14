# Repository Guidelines

## Project Structure & Module Organization

The Python package lives in `llmeval/`. Inference backends are under `llmeval/inference/` (`online.py`, `offline.py`, and `mc.py`); task-specific scoring is under `llmeval/tasks/`; shared configuration, logging, prompts, and retry policy are in `llmeval/utils/`. Tests mirror this layout in `tests/inference/`, `tests/tasks/`, and `tests/utils/`. Dataset preparation and distributed inference utilities live in `scripts/`. Runnable model configurations belong in `examples/`, while user-facing explanations belong in `docs/`. Benchmark JSONL inputs are stored in `data/`.

## Build, Test, and Development Commands

- `pip install -e .` installs the package for local development.
- `pip install -e '.[vllm]'` adds the optional GPU offline-inference backend.
- `pytest -q` runs the complete test suite.
- `pytest tests/inference/test_common.py -q` runs a focused test module.
- `ruff format llmeval tests scripts` formats Python code.
- `ruff check llmeval tests scripts` checks imports, style, and common bugs.
- `mypy llmeval --ignore-missing-imports` runs static type checks.
- `pre-commit run --all-files` executes the repository's full formatting and file-hygiene checks.

## Coding Style & Naming Conventions

Use Python 3.10-compatible syntax, four-space indentation, double quotes, and Ruff's 88-character line length. Prefer small, explicit functions and typed dataclasses for configuration. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_CASE` for constants. Keep task-specific behavior inside its Math, MC, or Code module and place genuinely shared behavior in `common.py` or `postprocess.py`. Preserve existing JSONL fields and resume semantics unless a migration is documented and tested.

## Testing Guidelines

Pytest discovers `test_*.py` files and `test_*` functions. Add focused regression tests for bug fixes, especially around malformed API responses, resume files, sample completeness, and metric denominators. Use `tmp_path` for filesystem tests and monkeypatch external APIs or model engines. No coverage threshold is enforced, but changed behavior should be exercised on success and failure paths.

## Commit & Pull Request Guidelines

Follow the existing Conventional Commit style: `fix: ...`, `feat(mc_eval): ...`, `refactor: ...`, or `chore: ...`; use `!` for breaking changes. Keep commits scoped and avoid mixing generated benchmark output with source changes. Pull requests should explain the behavioral change, affected tasks/backends, compatibility impact, and commands run. Include sample output or logs when metrics, JSON schemas, or scripts change.

## Security & Configuration Tips

Never commit API keys, tokens, model credentials, or personal absolute paths. Pass secrets through environment variables. Generated-code evaluation is unsafe by nature; keep `allow_unsafe_code` opt-in and run it only in an isolated environment.
