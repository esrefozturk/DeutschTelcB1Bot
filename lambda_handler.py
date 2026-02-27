"""
AWS Lambda entry point for the TELC B1 Telegram bot.

Flow:
  API Gateway POST /webhook  →  lambda_handler()
    → parses the Telegram Update JSON
    → dispatches through the same Application + handlers as bot.py
    → returns HTTP 200 immediately

Key design decisions:
  • A single asyncio event loop and Application instance are created at cold-start
    and reused across all warm invocations (no re-initialization overhead).
  • DynamoDB replaces SQLite for persistent state.
  • polling is NOT used — Telegram pushes updates via webhook.
"""

import asyncio
import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()  # no-op in Lambda (env vars come from the runtime), safe locally

from telegram import Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

import gemini_client
from database_dynamo import DynamoDatabase

# Import stateless handler functions from bot.py.
# bot.py imports `from database import Database` at the top — that is fine:
# it just imports the *class*, no SQLite file is opened until Database() is
# instantiated.  We never call bot.main() here, so no SQLite DB is created.
from bot import (
    cmd_help,
    cmd_next,
    cmd_pause,
    cmd_resume,
    cmd_start,
    cmd_stats,
    cmd_topic,
    on_callback,
    on_message,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# ── One-time cold-start initialization ───────────────────────────────────────
# Everything below runs once when Lambda loads the module (cold start).
# On warm invocations the module is already in memory — skip straight to
# lambda_handler().

gemini_client.init_gemini(GEMINI_API_KEY)
_db = DynamoDatabase()


async def _build_application() -> Application:
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.bot_data["db"] = _db

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("next",   cmd_next))
    app.add_handler(CommandHandler("pause",  cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("stats",  cmd_stats))
    app.add_handler(CommandHandler("topic",  cmd_topic))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    # initialize() sets up the bot client (fetches bot info, etc.)
    # We do NOT call start() — that would spin up the polling scheduler.
    await app.initialize()
    return app


# Create a persistent event loop so that the Application (and its underlying
# httpx client) are always used from the same loop across warm invocations.
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)
_application: Application = _loop.run_until_complete(_build_application())

logger.info("Application initialized (cold start complete).")

# ── Lambda handler ────────────────────────────────────────────────────────────


def lambda_handler(event: dict, context) -> dict:
    """
    Receives an API Gateway proxy event and dispatches the Telegram update.
    Always returns 200 — Telegram will retry on non-2xx, causing duplicate
    processing, so we absorb all errors here and log them instead.
    """
    body_raw = event.get("body") or ""
    if not body_raw:
        return _ok("no body")

    try:
        body = json.loads(body_raw)
    except json.JSONDecodeError:
        logger.error("Malformed JSON body: %.200s", body_raw)
        return _ok("bad json")

    _loop.run_until_complete(_process(body))
    return _ok("ok")


async def _process(body: dict) -> None:
    try:
        update = Update.de_json(body, _application.bot)
        await _application.process_update(update)
    except Exception:
        # Log but swallow — we must return 200 to Telegram to prevent
        # infinite retry loops.
        logger.exception("Unhandled error processing update: %s", body)


def _ok(msg: str = "OK") -> dict:
    return {"statusCode": 200, "body": msg}
