# CLAUDE.md — Project context for Claude Code

## What this project is

Adaptive TELC B1 German learning Telegram bot. Runs on AWS Lambda (webhook mode)
in production, and locally with Python polling mode. Uses Google Gemini 2.0 Flash
for question generation and answer evaluation.

## Key files

| File | Role |
|---|---|
| `bot.py` | All Telegram handlers + `main()` for local polling. **Handlers are stateless** — they get the DB via `context.bot_data["db"]`. |
| `lambda_handler.py` | Lambda entry point. Builds the same Application as `bot.py` but with DynamoDB and webhook. Single event loop persists across warm invocations. |
| `gemini_client.py` | Two async functions: `generate_question()` and `evaluate_answer()`. Both use `run_in_executor` to call the synchronous Gemini SDK without blocking the event loop. |
| `adaptive.py` | Topic taxonomy (`TOPICS` dict), `pick_next_params()` for weighted topic selection, `adjust_difficulty()`. No I/O — pure logic. |
| `database.py` | SQLite backend for local dev. |
| `database_dynamo.py` | DynamoDB backend for Lambda. **Same public interface** as `database.py` — swap by changing the import. |
| `template.yaml` | AWS SAM. Defines Lambda + API Gateway + 3 DynamoDB tables. Stage parameter (`prod`/`staging`) suffixes all resource names. |
| `deploy.sh` | Wraps `sam build` + `sam deploy` + `set_webhook.py`. Reads secrets from `.env`. |
| `set_webhook.py` | Pure stdlib — registers/removes Telegram webhook. |

## Architecture decisions

- **Webhook not polling in Lambda**: Lambda can't run a long-lived poll loop.
  Telegram pushes each update to the API Gateway URL.
- **Single persistent event loop in Lambda**: `_loop = asyncio.new_event_loop()`
  created at module load, reused across warm invocations so the httpx client
  inside python-telegram-bot isn't re-created on every call.
- **DynamoDB for state**: SQLite `/tmp` is ephemeral in Lambda. DynamoDB is
  serverless, durable, and requires no connection management.
- **Gemini JSON mode**: `response_mime_type="application/json"` is set so Gemini
  returns structured data. `_extract_json()` strips any stray markdown fences.
- **Always return 200 to Telegram**: errors are logged, never re-raised, to
  prevent Telegram's retry loop from reprocessing the same update.

## DynamoDB schema

### `DeutschBotUsers-prod`
PK: `user_id` (S)
Fields: `username`, `first_name`, `created_at`, `is_paused` (0/1), `last_active`, `last_reminder_sent`, `current_streak`, `last_streak_date`

### `DeutschBotSessions-prod`
PK: `user_id` (S)
Fields: `pending_question` (JSON string), `exam_state` (JSON string), `questions_sent`, `updated_at`, `ttl`

### `DeutschBotPerformance-prod`
PK: `user_id` (S), SK: `topic_subtopic` (S, format: `"grammar#cases"`)
Fields: `correct`, `incorrect` (ADD atomically), `total_score`, `difficulty` (Decimal 1–5), `review_interval` (SRS days), `last_tested`

## Adaptive engine

Weight formula: `base_weight × (1 + error_rate²) × srs_factor`
- Unseen subtopics default to 50% error rate so they get explored early
- Difficulty: starts at 2.0, +0.5 on correct, -0.75 on wrong, clamped [1.0, 5.0]
- Base weights: grammar=3, vocabulary=3, reading=2, writing=1.5
- SRS: review_interval doubles on correct (1→2→4→…→30d), resets to 1d on wrong
- srs_factor: `1.0 + min(overdue_days, 14) × 0.3` (never-tested = 2.0)

## Deployed stack (prod)

- Region: `us-east-1`
- Stack: `deutsch-telc-bot-prod`
- Lambda: `DeutschTelcBot-prod`
- Logs: `/aws/lambda/DeutschTelcBot-prod`

## Local dev

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in TELEGRAM_TOKEN + GEMINI_API_KEY
python bot.py         # polling mode, SQLite DB
```

## Redeploy after code changes

```bash
./deploy.sh --no-guided
```

## Secrets (never commit)

- `.env` is in `.gitignore`
- Secrets are passed to Lambda via CloudFormation parameter overrides (masked in logs)
- Do NOT hardcode tokens in any source file
