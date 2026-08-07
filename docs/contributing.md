# Contributing to PyDOE

Thank you for your interest in contributing to PyDOE! 
This guide explains how to report issues, propose changes, and maintain consistency in code, documentation, and testing.

## How to Contribute

You can contribute in several ways:

- Reporting bugs or suggesting improvements via GitHub Issues.
- Answering questions and helping other users.
- Submitting pull requests.
- Improving or expanding documentation.
- Adding or improving tests.
- Reviewing open pull requests. It is educational and productive.

> Before starting significant work (other than fixing small typos in documentation or small bug fixes), please open an issue to discuss your proposal.

## Reporting Issues

When filing an issue:

- Search existing issues to avoid duplicates.
- Use a clear and descriptive title starting with relevant keywords like `feature-req - `, `bug - ` wherever possible.
- Provide detailed steps to reproduce the problem (for bug reports only).
- Describe the expected and actual behavior.
- Include environment details (Python version, OS, dependencies etc.).
- If possible, provide a minimal reproducible example.

## Submitting Pull Requests

### Prerequisites

You will need following prerequisites for getting started with development:

- git
- [uv](https://docs.astral.sh/uv/getting-started/installation)
- any code editor/IDE of your choice like vscode, pycharm etc.

### Installation and Setup

Fork the repo on GitHub and clone it to your local machine.

```bash
git clone https://github.com/<your-username>/pydoe.git --branch main --depth 1
cd pydoe
uv sync
```

### Git Hooks

The repo ships a pre-commit hook in `.git-hooks/pre-commit` that runs
`ruff format --check`, `ruff check`, and the full test suite before each
commit, mirroring the CI checks. Enable it once per clone with:

```bash
git config core.hooksPath .git-hooks
```

If any of these checks fail, the commit is aborted so you can fix the
issue before committing.

### Commit Changes

Switch to a new branch for your changes.

```bash
git checkout -b <your-branch-name>
```

### Run Linter and Tests

Add and updata tests to ensure that tests cover your changes. Run tests and linter to ensure everything is consistent and working correctly.

```bash
# running linter and formatter
uvx ruff format .
uvx ruff check --fix .

# running tests
uv run pytest -n auto tests
```

### Build Documentation

It is recommeded to add documentation about your change using docstrings or editing docs files. If you have made any changes to the documentation, build and preview it locally. API docs are auto-generated from docstrings, however you need to add function manually to the relevant docs file. Be sure to check the warnings and errors while building the docs.

```bash
uv run python -m zensical build --clean --strict
```

### Open a Pull Request

After everything is finalized, commit your changes with a descriptive commit message, push your changes to your fork, and open a **DRAFT** pull request against the `main` branch of the parent repo. Be sure to provide a good description and mention any related issues.

Once your changes are ready for review, you can mark the PR as ready for review and some maintainer will review it.
