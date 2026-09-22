# Contributing to audiobook-tts

Thank you for your interest in contributing to **audiobook-tts**.

This project provides a production-grade, multi-device continuous audiobook generation engine. To keep the codebase robust, performant, and maintainable, please follow these principles.

---

## Issues are the Ledger

Open an issue before submitting a substantial pull request or architectural shift. 

Issues serve as the public ledger of architectural decisions and trade-offs. When reporting a bug or anomaly:

1. **Reproduction Input**: The exact text snippet or character sequence that triggered the issue.
2. **Hardware Configuration**: Devices provided to `--devices` (e.g. `cuda:1,cuda:0,cpu`), PyTorch version, and OS.
3. **Observed vs. Expected Output**: Verbatim terminal logs or error traces, and the expected audio behavior.

---

## Pull Request Guidelines

Before opening a pull request, verify the following:

### 1. Code Quality & Formatting
Ensure Python code meets standard styling and has no syntax or type errors:

```bash
# Verify Python syntax
python -m py_compile audiobook_tts/**/*.py

# Run ruff / flake8 linting if installed
ruff check audiobook_tts
```

### 2. Multi-Device Safety
- Do not introduce global mutable state into worker loops.
- Workers operate concurrently across heterogeneous hardware (GPUs and CPU cores). Inter-worker communication must remain strictly channel-based (`multiprocessing.Queue` or thread-safe primitives).
- Never block the SSE event stream or tie audio generation directly to HTTP request threads.

### 3. Commit Convention
Use concise [Conventional Commits](https://www.conventionalcommits.org/):
- `feat(...)`: New feature or capability
- `fix(...)`: Bug fix
- `perf(...)`: Performance optimization or throughput increase
- `docs(...)`: Documentation updates
- `refactor(...)`: Code cleanup without semantic change

All commit messages and pull request descriptions must be written in **English**.
