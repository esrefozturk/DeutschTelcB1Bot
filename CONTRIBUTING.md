# Contributing to DeutschTelcB1Bot

Thank you for your interest in contributing!

## Development setup

```bash
git clone https://github.com/esrefozturk/DeutschTelcB1Bot.git
cd DeutschTelcB1Bot

# Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt ruff

# Configure environment
cp .env.example .env
# Fill in TELEGRAM_TOKEN and GEMINI_API_KEY
```

Run locally (SQLite, no AWS needed):

```bash
python bot.py
```

## Code style

This project uses **[ruff](https://docs.astral.sh/ruff/)** for linting. Run it before pushing:

```bash
ruff check .
```

Configuration is in `ruff.toml`. The CI will fail if ruff reports errors.

## Making changes

1. **Branch from `main`** — use descriptive branch names like `fix/voice-handler`, `feat/new-question-type`, `chore/update-deps`.
2. **One topic per PR** — keep PRs focused so they're easier to review.
3. **Open a PR against `main`** — fill in the PR template.
4. **Wait for CI** — the `Lint & import check` job must pass.
5. **Wait for Cursor Bugbot** — review its findings:
   - If the issue is valid, fix it in a new commit on the same branch.
   - If the issue is a false positive, resolve the comment on GitHub with a brief explanation.
6. **Merge** — once CI and Bugbot are resolved, the PR can be merged. Merging to `main` triggers automatic deployment to AWS Lambda.

## Project structure

| File | Purpose |
|---|---|
| `bot.py` | Telegram command/message handlers; local polling entry point |
| `lambda_handler.py` | AWS Lambda entry point (webhook + EventBridge reminders) |
| `gemini_client.py` | Gemini API: question generation, evaluation, voice eval, hints |
| `adaptive.py` | Adaptive learning engine: topic selection, SRS, difficulty |
| `database.py` | SQLite backend (local dev) |
| `database_dynamo.py` | DynamoDB backend (production Lambda) |
| `template.yaml` | AWS SAM infrastructure (Lambda, API Gateway, DynamoDB, EventBridge) |
| `deploy.sh` | One-command manual build + deploy + webhook registration |

## Reporting issues

Please use the GitHub issue templates — bug reports and feature requests each have a form that collects the right information.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
