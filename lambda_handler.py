"""
AWS Lambda entry point for the TELC B1 Telegram bot.

Two trigger sources:
  1. API Gateway POST /webhook   — Telegram update (normal bot flow)
  2. EventBridge scheduled rules — reminder_type "inactivity" or "weekly"

Key design decisions:
  • A single asyncio event loop and Application instance are created at cold-start
    and reused across all warm invocations (no re-initialization overhead).
  • DynamoDB replaces SQLite for persistent state.
  • polling is NOT used — Telegram pushes updates via webhook.
"""

import asyncio
import html
import json
import logging
import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()  # no-op in Lambda (env vars come from the runtime), safe locally

import httpx
from telegram import Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

import gemini_client
from database_dynamo import DynamoDatabase

# Import stateless handler functions from bot.py.
# bot.py imports `from database import Database` at the top — that is fine:
# it just imports the *class*, no SQLite file is opened until Database() is
# instantiated.  We never call bot.main() here, so no SQLite DB is created.
from bot import (
    cmd_exam,
    cmd_help,
    cmd_next,
    cmd_pause,
    cmd_request_more_quota,
    cmd_start,
    cmd_stats,
    cmd_topic,
    on_callback,
    on_message,
    on_voice,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# How long without practice before we send a nudge, and how often we re-nudge
INACTIVITY_HOURS        = 4
REMINDER_COOLDOWN_HOURS = 24

# ── One-time cold-start initialization ───────────────────────────────────────

gemini_client.init_gemini(GEMINI_API_KEY)
_db = DynamoDatabase()


async def _build_application() -> Application:
    # Disable connection keep-alive so warm Lambda invocations never reuse a
    # TCP connection that AWS NAT has already silently dropped (→ ConnectTimeout).
    _request = HTTPXRequest(
        httpx_kwargs={"limits": httpx.Limits(max_connections=5, max_keepalive_connections=0)}
    )
    app = Application.builder().token(TELEGRAM_TOKEN).request(_request).build()
    app.bot_data["db"] = _db

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("next",   cmd_next))
    app.add_handler(CommandHandler("exam",   cmd_exam))
    app.add_handler(CommandHandler("stats",  cmd_stats))
    app.add_handler(CommandHandler("topic",  cmd_topic))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CommandHandler("pause",  cmd_pause))
    app.add_handler(CommandHandler("request_more_quota", cmd_request_more_quota))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))
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
    Handles both Telegram webhook events (from API Gateway) and scheduled
    EventBridge events for reminders.
    """
    # EventBridge scheduled event — no "body" key, has "reminder_type" instead
    reminder_type = event.get("reminder_type")
    if reminder_type:
        _loop.run_until_complete(_run_reminders(reminder_type))
        return _ok("reminders sent")

    # Normal Telegram webhook
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
        logger.exception("Unhandled error processing update: %s", body)


# ── Scheduled reminder logic ──────────────────────────────────────────────────


async def _run_reminders(reminder_type: str) -> None:
    bot  = _application.bot
    now  = datetime.now(timezone.utc)
    users = _db.get_all_users()
    logger.info("Running '%s' reminder for %d users", reminder_type, len(users))

    if reminder_type == "inactivity":
        for user in users:
            await _maybe_send_inactivity_nudge(bot, user, now)

    elif reminder_type == "weekly":
        for user in users:
            await _maybe_send_weekly_summary(bot, user)


async def _maybe_send_inactivity_nudge(bot, user: dict, now: datetime) -> None:
    uid = user["user_id"]

    if user.get("is_paused"):
        return  # user opted out of reminders

    last_active_str = user.get("last_active", "")
    if not last_active_str:
        return  # user has never been active

    try:
        last_active = datetime.fromisoformat(last_active_str)
        inactive_h  = (now - last_active).total_seconds() / 3600
        if inactive_h < INACTIVITY_HOURS:
            return  # still recently active

        # Respect cooldown — max one reminder per day
        last_reminder_str = user.get("last_reminder_sent", "")
        if last_reminder_str:
            last_reminder = datetime.fromisoformat(last_reminder_str)
            if (now - last_reminder).total_seconds() / 3600 < REMINDER_COOLDOWN_HOURS:
                return

        # Only show streak warning if they haven't already practiced today
        streak = user.get("current_streak", 0)
        today  = now.date().isoformat()
        already_practiced_today = last_active_str.startswith(today)
        streak_line = (
            f"\n\n⚠️ Don't break your <b>{streak}-day streak!</b>"
            if streak > 1 and not already_practiced_today else ""
        )
        await bot.send_message(
            chat_id=uid,
            text=(
                "👋 <b>Zeit zu üben!</b> You haven't practiced German in a while.\n\n"
                "A quick session now will keep your skills sharp. "
                f"Type /next for your next question! 🇩🇪{streak_line}\n\n"
                "<i>Don't want reminders? Type /pause to stop them.</i>"
            ),
            parse_mode="HTML",
        )
        _db.mark_reminder_sent(uid)
        logger.info("Sent inactivity nudge to user %s (inactive %.1fh)", uid, inactive_h)
    except Exception as exc:
        logger.warning("Inactivity nudge failed for user %s: %s", uid, exc)


async def _maybe_send_weekly_summary(bot, user: dict) -> None:
    uid = user["user_id"]
    if user.get("is_paused"):
        return
    try:
        s = _db.get_stats_summary(uid)
        if s["total_questions"] < 3:
            return  # not enough data for a meaningful summary

        lines = [
            "📅 <b>Your Weekly German Practice Check-in</b>",
            "",
            f"Questions answered (all time): <b>{s['total_questions']}</b>",
            f"Accuracy: <b>{s['accuracy']}%</b>  |  Avg score: <b>{s['avg_score']}%</b>",
            "",
        ]
        if s["weak_areas"]:
            lines.append("🔴 <b>Focus areas for this week:</b>")
            for w in s["weak_areas"]:
                er       = round(w["error_rate"] * 100)
                topic    = html.escape(w["topic"].replace("_", " "))
                subtopic = html.escape(w["subtopic"].replace("_", " "))
                lines.append(f"  • {topic} › {subtopic}  ({er}% errors)")
        else:
            lines.append("🟢 Great work — no major weak spots!")

        lines += ["", "Keep it up! Type /next to practice now. 💪"]

        await bot.send_message(
            chat_id=uid,
            text="\n".join(lines),
            parse_mode="HTML",
        )
        logger.info("Sent weekly summary to user %s", uid)
    except Exception as exc:
        logger.warning("Weekly summary failed for user %s: %s", uid, exc)


def _ok(msg: str = "OK") -> dict:
    return {"statusCode": 200, "body": msg}
