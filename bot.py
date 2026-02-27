"""
TELC B1 German Preparation Bot
--------------------------------
Commands:
  /start   – Welcome + start practice
  /next    – Get the next question immediately
  /pause   – Pause practice
  /resume  – Resume practice
  /stats   – Show personal performance statistics
  /topic   – List topics / focus on a specific topic
  /help    – Command reference

Any non-command message is treated as the answer to the pending question.
"""

import logging
import os
from textwrap import dedent
from typing import Optional

from dotenv import load_dotenv
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from adaptive import (
    TOPICS,
    adjust_difficulty,
    pick_next_params,
)
from database import Database
from gemini_client import evaluate_answer, explain_further, generate_question, init_gemini

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
GEMINI_API_KEY  = os.environ["GEMINI_API_KEY"]
DATABASE_PATH   = os.getenv("DATABASE_PATH", "deutsch_bot.db")

# ── Helpers ───────────────────────────────────────────────────────────────────

WELCOME_MSG = dedent("""\
    👋 *Willkommen!* I'm your personal TELC B1 German coach.

    I'll send you adaptive questions covering:
    • Grammar (cases, verbs, prepositions, …)
    • Vocabulary (daily life, work, travel, …)
    • Reading comprehension
    • Writing tasks

    Questions are chosen based on your past answers, so the areas you \
    struggle with will appear more often — keeping you growing.

    Type /next (or just answer) to begin. Viel Erfolg! 🇩🇪
""")

HELP_MSG = dedent("""\
    *Commands*

    /start   – Welcome message
    /next    – Get the next question now
    /pause   – Pause practice
    /resume  – Resume practice
    /stats   – Your performance summary
    /topic   – Browse or focus on a topic
    /help    – This message

    *Answering*
    • For multiple-choice questions tap a button OR type A / B / C / D.
    • For all other types just type your answer freely.
    • Minor spelling mistakes are OK; grammar concepts are graded strictly.
""")


def _difficulty_stars(d: float) -> str:
    filled = round(d)
    return "★" * filled + "☆" * (5 - filled)


def _build_mc_keyboard(options: list[str]) -> InlineKeyboardMarkup:
    """One button per option, each on its own row."""
    buttons = [
        [InlineKeyboardButton(opt, callback_data=f"answer:{opt[0]}")]
        for opt in options
    ]
    return InlineKeyboardMarkup(buttons)


def _action_keyboard() -> InlineKeyboardMarkup:
    """Buttons shown after evaluation: explain or continue."""
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("💡 Explain More", callback_data="explain"),
        InlineKeyboardButton("➡️ Next Question", callback_data="next_q"),
    ]])


def _fmt_question(q: dict, is_first: bool = False) -> str:
    """Format a question dict into Telegram markdown."""
    topic_label    = TOPICS.get(q["topic"], {}).get("label", q["topic"])
    subtopic_label = TOPICS.get(q["topic"], {}).get("subtopics", {}).get(q["subtopic"], q["subtopic"])
    stars          = _difficulty_stars(q.get("difficulty", 2))
    qtype_pretty   = q["question_type"].replace("_", " ").title()

    divider = "" if is_first else "〰️〰️〰️ *Next Question* 〰️〰️〰️\n\n"
    header = (
        f"{divider}"
        f"📚 *{topic_label}* › _{subtopic_label}_\n"
        f"Difficulty: {stars}  |  Type: {qtype_pretty}\n\n"
    )

    # Replace blank marker with a visible monospace placeholder.
    # Plain "___" gets swallowed by Telegram's Markdown italic parser.
    body = q["question"].replace("___", "`______`")

    if q["question_type"] == "sentence_building":
        body += "\n\n_(Arrange the words into a correct German sentence.)_"
    elif q["question_type"] == "fill_blank":
        body += "\n\n_(Fill in the blank.)_"
    elif q["question_type"] == "error_correction":
        body += "\n\n_(Find and correct the single grammatical error.)_"

    return header + body


async def _send_question(
    db: Database,
    user_id: int,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    forced_topic: Optional[str] = None,
    forced_subtopic: Optional[str] = None,
    is_first: bool = False,
):
    """Generate the next question and send it to the user."""
    performance = db.get_all_performance(user_id)

    if forced_topic and forced_subtopic:
        from adaptive import QUESTION_TYPES_BY_TOPIC
        import random
        topic    = forced_topic
        subtopic = forced_subtopic
        q_type   = random.choice(QUESTION_TYPES_BY_TOPIC[topic])
        perf_row = db.get_topic_performance(user_id, topic, subtopic)
        difficulty = perf_row.get("difficulty", 2.0)
    else:
        topic, subtopic, q_type, difficulty = pick_next_params(performance)

    # Give Gemini brief context to avoid immediately repeating the same question
    recent_ctx = None
    pending = db.get_pending_question(user_id)
    if pending:
        recent_ctx = f"{pending.get('topic')}/{pending.get('subtopic')}"

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        question = await generate_question(topic, subtopic, q_type, difficulty, recent_ctx)
    except Exception as exc:
        logger.error("generate_question failed: %s", exc)
        await context.bot.send_message(
            chat_id=chat_id,
            text="⚠️ Couldn't generate a question right now. Try /next in a moment.",
        )
        return

    db.set_pending_question(user_id, question)

    text    = _fmt_question(question, is_first=is_first)
    reply_markup = None
    if question["question_type"] == "multiple_choice" and "options" in question:
        reply_markup = _build_mc_keyboard(question["options"])
        # Append options to message text for clarity
        options_text = "\n".join(question["options"])
        text += f"\n\n{options_text}"

    await context.bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup,
    )


async def _handle_answer(
    db: Database,
    user_id: int,
    chat_id: int,
    user_answer: str,
    context: ContextTypes.DEFAULT_TYPE,
):
    """Evaluate the user's answer, give feedback, then send the next question."""
    pending = db.get_pending_question(user_id)
    if not pending:
        await _send_question(db, user_id, chat_id, context)
        return

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        result = await evaluate_answer(pending, user_answer)
    except Exception as exc:
        logger.error("evaluate_answer failed: %s", exc)
        await context.bot.send_message(
            chat_id=chat_id,
            text="⚠️ Couldn't evaluate your answer. Let me send a new question.",
        )
        await _send_question(db, user_id, chat_id, context)
        return

    is_correct = result.get("is_correct", False)
    score      = result.get("score", 0.0)
    feedback   = result.get("feedback", "")
    correction = result.get("correction", "")

    # Emoji header
    if score >= 0.9:
        header = "✅ *Correct!*"
    elif score >= 0.5:
        header = "🟡 *Partially correct*"
    else:
        header = "❌ *Not quite*"

    parts = [header]
    if feedback:
        parts.append(feedback)
    if correction:
        parts.append(f"*Correction:* {correction}")
    parts.append(f"*Correct answer:* `{pending.get('correct_answer', '')}`")
    if pending.get("explanation"):
        parts.append(f"💡 _{pending['explanation']}_")

    # Update DB
    old_perf   = db.get_topic_performance(user_id, pending["topic"], pending["subtopic"])
    new_diff   = adjust_difficulty(old_perf.get("difficulty", 2.0), is_correct)
    db.update_performance(user_id, pending["topic"], pending["subtopic"], is_correct, score, new_diff)

    # Keep pending question with answer context for "Explain More"
    pending["_answered"]      = True
    pending["_user_answer"]   = user_answer
    pending["_last_feedback"] = feedback + (f"\nCorrection: {correction}" if correction else "")
    pending["_explain_depth"] = 0
    db.set_pending_question(user_id, pending)

    await context.bot.send_message(
        chat_id=chat_id,
        text="\n\n".join(parts),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=_action_keyboard(),
    )


# ── Command handlers ──────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db: Database = context.bot_data["db"]
    user = update.effective_user
    db.upsert_user(user.id, user.username or "", user.first_name or "")
    db.set_paused(user.id, False)

    await update.message.reply_text(WELCOME_MSG, parse_mode=ParseMode.MARKDOWN)
    await _send_question(db, user.id, update.effective_chat.id, context, is_first=True)


async def cmd_next(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db: Database = context.bot_data["db"]
    user = update.effective_user
    db.upsert_user(user.id, user.username or "", user.first_name or "")
    db.set_paused(user.id, False)

    # If there is a pending question, skip it (don't evaluate)
    db.set_pending_question(user.id, None)
    await _send_question(db, user.id, update.effective_chat.id, context, is_first=True)


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db: Database = context.bot_data["db"]
    db.set_paused(update.effective_user.id, True)
    await update.message.reply_text(
        "⏸ Practice paused. Type /resume whenever you're ready."
    )


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db: Database = context.bot_data["db"]
    user = update.effective_user
    db.set_paused(user.id, False)
    await update.message.reply_text("▶️ Let's go!")
    await _send_question(db, user.id, update.effective_chat.id, context, is_first=True)


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db: Database = context.bot_data["db"]
    user_id = update.effective_user.id
    s = db.get_stats_summary(user_id)

    if s["total_questions"] == 0:
        await update.message.reply_text(
            "No stats yet — answer a few questions first! Type /next to start."
        )
        return

    lines = [
        "📊 *Your Progress*",
        "",
        f"Questions answered: *{s['total_questions']}*",
        f"Correct: *{s['total_correct']}*  |  Wrong: *{s['total_incorrect']}*",
        f"Accuracy: *{s['accuracy']}%*  |  Avg score: *{s['avg_score']}%*",
        "",
        "_Accuracy counts fully correct answers only. Avg score includes partial credit._",
        "",
    ]

    if s["weak_areas"]:
        lines.append("🔴 *Areas to focus on:*")
        for w in s["weak_areas"]:
            er       = round(w["error_rate"] * 100)
            avg      = round(w.get("avg_score", 0) * 100)
            topic    = w["topic"].replace("_", " ")
            subtopic = w["subtopic"].replace("_", " ")
            lines.append(f"  • {topic} › {subtopic}  ({er}% errors, {avg}% avg score)")
    else:
        lines.append("🟢 No major weak spots yet — keep going!")

    await update.message.reply_text(
        "\n".join(lines), parse_mode=ParseMode.MARKDOWN
    )


async def cmd_topic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db: Database = context.bot_data["db"]
    args = context.args

    if not args:
        # Show topic list
        lines = ["*Available topics* (use `/topic <name>` to focus):\n"]
        for key, data in TOPICS.items():
            subtopics = ", ".join(data["subtopics"].keys())
            lines.append(f"*{key}*: {subtopics}")
        await update.message.reply_text(
            "\n".join(lines), parse_mode=ParseMode.MARKDOWN
        )
        return

    topic_arg = args[0].lower()
    subtopic_arg = args[1].lower() if len(args) > 1 else None

    if topic_arg not in TOPICS:
        await update.message.reply_text(
            f"Unknown topic '{topic_arg}'. Use /topic to see available topics."
        )
        return

    if subtopic_arg and subtopic_arg not in TOPICS[topic_arg]["subtopics"]:
        available = ", ".join(TOPICS[topic_arg]["subtopics"].keys())
        await update.message.reply_text(
            f"Unknown subtopic. Available for *{topic_arg}*: {available}",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    if subtopic_arg is None:
        import random
        subtopic_arg = random.choice(list(TOPICS[topic_arg]["subtopics"].keys()))

    await update.message.reply_text(
        f"Focusing on *{topic_arg}* › _{subtopic_arg}_",
        parse_mode=ParseMode.MARKDOWN,
    )
    db.set_pending_question(update.effective_user.id, None)
    await _send_question(
        db,
        update.effective_user.id,
        update.effective_chat.id,
        context,
        forced_topic=topic_arg,
        forced_subtopic=subtopic_arg,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_MSG, parse_mode=ParseMode.MARKDOWN)


# ── Message handler (free-text answers) ──────────────────────────────────────

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db: Database = context.bot_data["db"]
    user = update.effective_user
    db.upsert_user(user.id, user.username or "", user.first_name or "")

    row = db.get_user(user.id)
    if row and row["is_paused"]:
        await update.message.reply_text(
            "Practice is paused. Type /resume to continue."
        )
        return

    pending = db.get_pending_question(user.id)
    if pending and pending.get("_answered"):
        await update.message.reply_text(
            "Use the buttons above — *💡 Explain More* or *➡️ Next Question*.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    user_text = update.message.text.strip()
    await _handle_answer(db, user.id, update.effective_chat.id, user_text, context)


# ── Callback handler (inline keyboard answers) ────────────────────────────────

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    db: Database = context.bot_data["db"]
    user = query.from_user
    db.upsert_user(user.id, user.username or "", user.first_name or "")

    data = query.data  # e.g. "answer:A", "explain", "next_q"

    if data.startswith("answer:"):
        user_answer = data.split(":", 1)[1]
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        await _handle_answer(
            db, user.id, query.message.chat_id, user_answer, context
        )

    elif data == "next_q":
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        db.set_pending_question(user.id, None)
        await _send_question(db, user.id, query.message.chat_id, context, is_first=True)

    elif data == "explain":
        pending = db.get_pending_question(user.id)
        if not pending or not pending.get("_answered"):
            await query.answer("No question context found.")
            return
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        depth = pending.get("_explain_depth", 0)
        try:
            await context.bot.send_chat_action(chat_id=query.message.chat_id, action="typing")
            explanation = await explain_further(
                pending,
                pending.get("_user_answer", ""),
                pending.get("_last_feedback", ""),
                depth,
            )
        except Exception as exc:
            logger.error("explain_further failed: %s", exc)
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="⚠️ Couldn't generate explanation right now.",
                reply_markup=_action_keyboard(),
            )
            return
        pending["_explain_depth"] = depth + 1
        pending["_last_feedback"] = explanation[:500]  # keep context trimmed
        db.set_pending_question(user.id, pending)
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=explanation,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=_action_keyboard(),
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    init_gemini(GEMINI_API_KEY)
    db = Database(DATABASE_PATH)

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.bot_data["db"] = db

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("next",   cmd_next))
    app.add_handler(CommandHandler("pause",  cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("stats",  cmd_stats))
    app.add_handler(CommandHandler("topic",  cmd_topic))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    logger.info("Bot is running…")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
