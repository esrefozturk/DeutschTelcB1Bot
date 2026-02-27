"""
TELC B1 German Preparation Bot
--------------------------------
Commands:
  /start   – Welcome + start practice
  /next    – Get the next question immediately
  /exam    – Start a 20-question TELC B1 practice exam
  /stats   – Show personal performance statistics
  /topic   – List topics / focus on a specific topic
  /help    – Command reference

Any non-command message is treated as the answer to the pending question.
"""

import html
import logging
import os
import random
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
    QUESTION_TYPES_BY_TOPIC,
    TOPICS,
    adjust_difficulty,
    next_review_interval,
    pick_next_params,
)
from database import Database
from gemini_client import GeminiQuotaExceeded, evaluate_answer, evaluate_voice_answer, explain_further, generate_question, get_hint, init_gemini

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

# ── Static messages (HTML) ────────────────────────────────────────────────────

WELCOME_MSG = (
    "👋 <b>Willkommen!</b> I'm your personal TELC B1 German coach.\n\n"
    "I'll send you adaptive questions covering:\n"
    "• Grammar (cases, verbs, prepositions, …)\n"
    "• Vocabulary (daily life, work, travel, …)\n"
    "• Reading comprehension\n"
    "• Writing tasks\n\n"
    "Questions use spaced repetition — topics due for review surface more often, "
    "and your streak grows every day you practice.\n\n"
    "Type /next to begin, or /exam for a full practice test. Viel Erfolg! 🇩🇪"
)

HELP_MSG = (
    "<b>Commands</b>\n\n"
    "/start  – Welcome message\n"
    "/next   – Get the next question now\n"
    "/exam   – 20-question TELC B1 practice test\n"
    "/stats  – Your performance summary\n"
    "/topic  – Browse or focus on a topic\n"
    "/help   – This message\n\n"
    "<b>Answering</b>\n"
    "• For multiple-choice questions tap a button OR type A / B / C / D.\n"
    "• For all other types just type your answer freely.\n"
    "• You can also send a <b>voice message</b> 🎤 — it will be transcribed and evaluated.\n"
    "• Minor spelling mistakes are OK; grammar concepts are graded strictly."
)

# Streak milestone messages
_STREAK_MILESTONES = {
    3:   "🌱 <b>3-day streak!</b> You're building a habit!",
    7:   "⭐ <b>7-day streak!</b> One full week — Wunderbar!",
    14:  "🌟 <b>14-day streak!</b> Two weeks of German!",
    30:  "🏆 <b>30-day streak!</b> An entire month — incredible!",
    50:  "💎 <b>50-day streak!</b> Unstoppable!",
    100: "🔱 <b>100-day streak!</b> You're a legend!",
}

# Exam question distribution across topics
_EXAM_DISTRIBUTION = [
    ("grammar",    8),
    ("vocabulary", 6),
    ("reading",    4),
    ("writing",    2),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _esc(text: str) -> str:
    return html.escape(str(text))


def _difficulty_stars(d: float) -> str:
    filled = round(d)
    return "★" * filled + "☆" * (5 - filled)


_MAX_HINTS = 2


def _question_keyboard(question: dict, hint_count: int = 0) -> Optional[InlineKeyboardMarkup]:
    """
    Keyboard attached to a question before the user answers.
    - Multiple-choice: one button per option + hint button below.
    - All other types: just the hint button.
    No hint button in exam mode (caller passes hint_count=_MAX_HINTS to suppress it).
    """
    rows = []
    if question["question_type"] == "multiple_choice" and "options" in question:
        for opt in question["options"]:
            rows.append([InlineKeyboardButton(opt, callback_data=f"answer:{opt[0]}")])
    if hint_count < _MAX_HINTS:
        label = "🔍 Get Hint" if hint_count == 0 else "🔍 Another Hint"
        rows.append([InlineKeyboardButton(label, callback_data="hint")])
    return InlineKeyboardMarkup(rows) if rows else None


def _action_keyboard() -> InlineKeyboardMarkup:
    """Buttons after normal evaluation."""
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("💡 Explain More", callback_data="explain"),
        InlineKeyboardButton("➡️ Next Question", callback_data="next_q"),
    ]])


def _exam_keyboard() -> InlineKeyboardMarkup:
    """Button after exam question evaluation — no explanation during exam."""
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("➡️ Next Question", callback_data="exam_next"),
    ]])


def _fmt_question(q: dict, is_first: bool = False, progress: str = "") -> str:
    topic_label    = TOPICS.get(q["topic"], {}).get("label", q["topic"])
    subtopic_label = TOPICS.get(q["topic"], {}).get("subtopics", {}).get(q["subtopic"], q["subtopic"])
    stars          = _difficulty_stars(q.get("difficulty", 2))
    qtype_pretty   = q["question_type"].replace("_", " ").title()

    if progress:
        divider = f"📝 <b>{_esc(progress)}</b>\n\n"
    elif is_first:
        divider = ""
    else:
        divider = "〰️〰️〰️ <b>Next Question</b> 〰️〰️〰️\n\n"

    header = (
        f"{divider}"
        f"📚 <b>{_esc(topic_label)}</b> › <i>{_esc(subtopic_label)}</i>\n"
        f"Difficulty: {stars}  |  Type: {_esc(qtype_pretty)}\n\n"
    )

    body = _esc(q["question"]).replace("___", "<code>______</code>")

    if q["question_type"] == "sentence_building":
        body += "\n\n<i>(Arrange the words into a correct German sentence.)</i>"
    elif q["question_type"] == "fill_blank":
        body += "\n\n<i>(Fill in the blank.)</i>"
    elif q["question_type"] == "error_correction":
        body += "\n\n<i>(Find and correct the single grammatical error.)</i>"

    return header + body


def _build_exam_plan() -> dict:
    """Build a balanced 20-question exam plan across all topic areas."""
    plan = []
    for topic, count in _EXAM_DISTRIBUTION:
        subtopics = list(TOPICS[topic]["subtopics"].keys())
        for i in range(count):
            subtopic = subtopics[i % len(subtopics)]
            q_type   = random.choice(QUESTION_TYPES_BY_TOPIC[topic])
            plan.append({
                "topic":      topic,
                "subtopic":   subtopic,
                "q_type":     q_type,
                "difficulty": 2.5,  # standard B1 level for exam
            })
    random.shuffle(plan)
    return {"active": True, "total": 20, "done": 0, "plan": plan, "results": []}


# ── Core question / answer logic ──────────────────────────────────────────────

async def _send_question(
    db: Database,
    user_id: int,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    forced_topic: Optional[str] = None,
    forced_subtopic: Optional[str] = None,
    is_first: bool = False,
):
    performance = db.get_all_performance(user_id)

    if forced_topic and forced_subtopic:
        topic    = forced_topic
        subtopic = forced_subtopic
        q_type   = random.choice(QUESTION_TYPES_BY_TOPIC[topic])
        perf_row = db.get_topic_performance(user_id, topic, subtopic)
        difficulty = perf_row.get("difficulty", 2.0)
    else:
        topic, subtopic, q_type, difficulty = pick_next_params(performance)

    recent_ctx = None
    pending = db.get_pending_question(user_id)
    if pending:
        recent_ctx = f"{pending.get('topic')}/{pending.get('subtopic')}"

    avoided = db.get_recent_questions(user_id)

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        question = await generate_question(topic, subtopic, q_type, difficulty, recent_ctx, avoided or None)
    except GeminiQuotaExceeded:
        await context.bot.send_message(
            chat_id=chat_id,
            text="📵 <b>Daily AI quota reached.</b> The free tier resets at midnight UTC. Try again then!",
            parse_mode=ParseMode.HTML,
        )
        return
    except Exception as exc:
        logger.error("generate_question failed: %s", exc)
        await context.bot.send_message(
            chat_id=chat_id,
            text="⚠️ Couldn't generate a question right now. Try /next in a moment.",
        )
        return

    text         = _fmt_question(question, is_first=is_first)
    reply_markup = _question_keyboard(question)  # MC buttons + hint
    if question["question_type"] == "multiple_choice" and "options" in question:
        options_text = "\n".join(_esc(opt) for opt in question["options"])
        text += f"\n\n{options_text}"

    msg = await context.bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup,
    )
    # Store message_id so _handle_answer can remove the hint button after a text answer.
    if reply_markup:
        question["_question_msg_id"] = msg.message_id
    db.set_pending_question(user_id, question, count=True)
    db.add_recent_question(user_id, question.get("question", ""))


async def _send_exam_question(
    db: Database,
    user_id: int,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    exam_state: dict,
):
    done  = exam_state["done"]
    total = exam_state["total"]
    plan  = exam_state["plan"]

    if done >= total:
        await _send_exam_summary(db, user_id, chat_id, context, exam_state)
        return

    params     = plan[done]
    topic      = params["topic"]
    subtopic   = params["subtopic"]
    q_type     = params["q_type"]
    difficulty = params["difficulty"]
    progress   = f"Question {done + 1} of {total}"

    avoided = db.get_recent_questions(user_id)

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        question = await generate_question(topic, subtopic, q_type, difficulty, None, avoided or None)
    except GeminiQuotaExceeded:
        await context.bot.send_message(
            chat_id=chat_id,
            text="📵 <b>Daily AI quota reached.</b> The free tier resets at midnight UTC. Try again then!",
            parse_mode=ParseMode.HTML,
        )
        return
    except Exception as exc:
        logger.error("exam generate_question failed: %s", exc)
        await context.bot.send_message(
            chat_id=chat_id,
            text="⚠️ Couldn't generate question. Try /exam to restart.",
        )
        return

    question["_exam_mode"] = True

    text         = _fmt_question(question, is_first=True, progress=progress)
    # Exam mode: MC buttons only, no hint button (_MAX_HINTS suppresses it)
    reply_markup = _question_keyboard(question, hint_count=_MAX_HINTS)
    if question["question_type"] == "multiple_choice" and "options" in question:
        options_text = "\n".join(_esc(opt) for opt in question["options"])
        text += f"\n\n{options_text}"

    await context.bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup,
    )
    db.set_pending_question(user_id, question, count=True)
    db.add_recent_question(user_id, question.get("question", ""))


async def _send_exam_summary(
    db: Database,
    user_id: int,
    chat_id: int,
    context: ContextTypes.DEFAULT_TYPE,
    exam_state: dict,
):
    results   = exam_state.get("results", [])
    total     = len(results)
    if total == 0:
        await context.bot.send_message(chat_id=chat_id, text="No results to summarise.")
        db.set_exam_state(user_id, None)
        db.set_pending_question(user_id, None)
        return

    correct   = sum(1 for r in results if r["is_correct"])
    avg_score = sum(r["score"] for r in results) / total

    grade = (
        "A" if avg_score >= 0.90 else
        "B" if avg_score >= 0.75 else
        "C" if avg_score >= 0.60 else
        "D"
    )

    topic_stats: dict = {}
    for r in results:
        t = r["topic"]
        if t not in topic_stats:
            topic_stats[t] = {"correct": 0, "total": 0, "score": 0.0}
        topic_stats[t]["total"]   += 1
        topic_stats[t]["correct"] += int(r["is_correct"])
        topic_stats[t]["score"]   += r["score"]

    lines = [
        "📝 <b>Exam Complete!</b>",
        "",
        f"Score: <b>{correct}/{total}</b>  ({round(avg_score * 100)}%)  Grade: <b>{grade}</b>",
        "",
        "<b>Topic breakdown:</b>",
    ]
    for topic, stats in topic_stats.items():
        label   = _esc(TOPICS[topic]["label"])
        t_score = round(stats["score"] / stats["total"] * 100)
        lines.append(f"  • {label}: {stats['correct']}/{stats['total']} ({t_score}%)")

    if avg_score >= 0.80:
        lines += ["", "🎉 Excellent work! You're well prepared for TELC B1!"]
    elif avg_score >= 0.60:
        lines += ["", "👍 Good effort! Keep practicing your weak areas."]
    else:
        lines += ["", "💪 Keep going — regular practice will get you there!"]

    lines += ["", "Type /next to continue, or /exam for another test."]

    db.set_exam_state(user_id, None)
    db.set_pending_question(user_id, None)

    await context.bot.send_message(
        chat_id=chat_id,
        text="\n".join(lines),
        parse_mode=ParseMode.HTML,
    )


async def _handle_answer(
    db: Database,
    user_id: int,
    chat_id: int,
    user_answer: str,
    context: ContextTypes.DEFAULT_TYPE,
):
    pending = db.get_pending_question(user_id)
    if not pending:
        await _send_question(db, user_id, chat_id, context)
        return

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        result = await evaluate_answer(pending, user_answer)
    except GeminiQuotaExceeded:
        await context.bot.send_message(
            chat_id=chat_id,
            text="📵 <b>Daily AI quota reached.</b> The free tier resets at midnight UTC. Try again then!",
            parse_mode=ParseMode.HTML,
        )
        return
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

    if score >= 0.9:
        header = "✅ <b>Correct!</b>"
    elif score >= 0.5:
        header = "🟡 <b>Partially correct</b>"
    else:
        header = "❌ <b>Not quite</b>"

    parts = [header]
    if feedback:
        parts.append(_esc(feedback))
    if correction:
        parts.append(f"<b>Correction:</b> {_esc(correction)}")
    parts.append(f"<b>Correct answer:</b> <code>{_esc(pending.get('correct_answer', ''))}</code>")
    if pending.get("explanation"):
        parts.append(f"💡 <i>{_esc(pending['explanation'])}</i>")

    # Update performance (SRS + difficulty)
    old_perf    = db.get_topic_performance(user_id, pending["topic"], pending["subtopic"])
    new_diff    = adjust_difficulty(old_perf.get("difficulty", 2.0), is_correct)
    new_ri      = next_review_interval(old_perf.get("review_interval", 1.0), is_correct)
    db.update_performance(
        user_id, pending["topic"], pending["subtopic"],
        is_correct, score, new_diff, new_ri,
    )

    # Update streak (only in regular practice mode, not exam)
    is_exam = pending.get("_exam_mode", False)
    if not is_exam:
        streak, is_new_day = db.update_streak(user_id)
        if is_new_day:
            milestone_msg = _STREAK_MILESTONES.get(streak)
            if milestone_msg:
                parts.append(milestone_msg)
            elif streak > 1:
                parts.append(f"🔥 <b>{streak}-day streak!</b>")

    # Remove the hint button (or MC buttons) from the question message when the
    # user answers by typing — button-based MC answers already handle this via
    # edit_message_reply_markup in on_callback, so failures here are silently ignored.
    msg_id = pending.get("_question_msg_id")
    if msg_id:
        try:
            await context.bot.edit_message_reply_markup(
                chat_id=chat_id, message_id=msg_id, reply_markup=None
            )
        except Exception:
            pass

    # Choose keyboard based on mode
    if is_exam:
        keyboard = _exam_keyboard()
        # Update exam state
        exam_state = db.get_exam_state(user_id)
        if exam_state:
            exam_state["done"] += 1
            exam_state["results"].append({
                "topic":      pending["topic"],
                "subtopic":   pending["subtopic"],
                "is_correct": is_correct,
                "score":      score,
            })
            db.set_exam_state(user_id, exam_state)
    else:
        keyboard = _action_keyboard()

    # Keep pending with answer context for "Explain More"
    pending["_answered"]      = True
    pending["_user_answer"]   = user_answer
    pending["_last_feedback"] = feedback + (f"\nCorrection: {correction}" if correction else "")
    pending["_explain_depth"] = 0
    db.set_pending_question(user_id, pending)

    await context.bot.send_message(
        chat_id=chat_id,
        text="\n\n".join(parts),
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )


# ── Command handlers ──────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db: Database = context.bot_data["db"]
    user = update.effective_user
    db.upsert_user(user.id, user.username or "", user.first_name or "")
    db.set_exam_state(user.id, None)

    await update.message.reply_text(WELCOME_MSG, parse_mode=ParseMode.HTML)
    await _send_question(db, user.id, update.effective_chat.id, context, is_first=True)


async def cmd_next(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db: Database = context.bot_data["db"]
    user = update.effective_user
    db.upsert_user(user.id, user.username or "", user.first_name or "")
    db.set_exam_state(user.id, None)
    db.set_pending_question(user.id, None)
    await _send_question(db, user.id, update.effective_chat.id, context, is_first=True)


async def cmd_exam(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db: Database = context.bot_data["db"]
    user = update.effective_user
    db.upsert_user(user.id, user.username or "", user.first_name or "")

    exam_state = _build_exam_plan()
    db.set_exam_state(user.id, exam_state)
    db.set_pending_question(user.id, None)

    await update.message.reply_text(
        "📝 <b>Exam Mode</b> — 20 questions\n\n"
        "This simulates a TELC B1 practice test across all topic areas. "
        "No explanations during the exam — focus and answer!\n\n"
        "Good luck! Viel Erfolg! 🍀",
        parse_mode=ParseMode.HTML,
    )
    await _send_exam_question(db, user.id, update.effective_chat.id, context, exam_state)


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db: Database = context.bot_data["db"]
    user_id = update.effective_user.id
    s = db.get_stats_summary(user_id)

    if s["total_questions"] == 0:
        await update.message.reply_text(
            "No stats yet — answer a few questions first! Type /next to start."
        )
        return

    streak = s.get("streak", 0)
    streak_line = f"🔥 Current streak: <b>{streak} day{'s' if streak != 1 else ''}</b>" if streak > 0 else ""

    lines = [
        "📊 <b>Your Progress</b>",
        "",
        f"Questions answered: <b>{s['total_questions']}</b>",
        f"Correct: <b>{s['total_correct']}</b>  |  Wrong: <b>{s['total_incorrect']}</b>",
        f"Accuracy: <b>{s['accuracy']}%</b>  |  Avg score: <b>{s['avg_score']}%</b>",
    ]
    if streak_line:
        lines.append(streak_line)
    lines += [
        "",
        "<i>Accuracy = fully correct answers. Avg score includes partial credit.</i>",
        "",
    ]

    if s["weak_areas"]:
        lines.append("🔴 <b>Areas to focus on:</b>")
        for w in s["weak_areas"]:
            er       = round(w["error_rate"] * 100)
            avg      = round(w.get("avg_score", 0) * 100)
            topic    = _esc(w["topic"].replace("_", " "))
            subtopic = _esc(w["subtopic"].replace("_", " "))
            lines.append(f"  • {topic} › {subtopic}  ({er}% errors, {avg}% avg score)")
    else:
        lines.append("🟢 No major weak spots yet — keep going!")

    await update.message.reply_text(
        "\n".join(lines), parse_mode=ParseMode.HTML
    )


async def cmd_topic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db: Database = context.bot_data["db"]
    args = context.args

    if not args:
        lines = ["<b>Available topics</b> (use <code>/topic &lt;name&gt;</code> to focus):\n"]
        for key, data in TOPICS.items():
            subtopics = ", ".join(data["subtopics"].keys())
            lines.append(f"<b>{_esc(key)}</b>: {_esc(subtopics)}")
        await update.message.reply_text(
            "\n".join(lines), parse_mode=ParseMode.HTML
        )
        return

    topic_arg    = args[0].lower()
    subtopic_arg = args[1].lower() if len(args) > 1 else None

    if topic_arg not in TOPICS:
        await update.message.reply_text(
            f"Unknown topic '{_esc(topic_arg)}'. Use /topic to see available topics.",
            parse_mode=ParseMode.HTML,
        )
        return

    if subtopic_arg and subtopic_arg not in TOPICS[topic_arg]["subtopics"]:
        available = ", ".join(TOPICS[topic_arg]["subtopics"].keys())
        await update.message.reply_text(
            f"Unknown subtopic. Available for <b>{_esc(topic_arg)}</b>: {_esc(available)}",
            parse_mode=ParseMode.HTML,
        )
        return

    if subtopic_arg is None:
        subtopic_arg = random.choice(list(TOPICS[topic_arg]["subtopics"].keys()))

    await update.message.reply_text(
        f"Focusing on <b>{_esc(topic_arg)}</b> › <i>{_esc(subtopic_arg)}</i>",
        parse_mode=ParseMode.HTML,
    )
    db.set_exam_state(update.effective_user.id, None)
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
    await update.message.reply_text(HELP_MSG, parse_mode=ParseMode.HTML)


# ── Message handler (free-text answers) ──────────────────────────────────────

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db: Database = context.bot_data["db"]
    user = update.effective_user
    db.upsert_user(user.id, user.username or "", user.first_name or "")

    pending = db.get_pending_question(user.id)
    if pending and pending.get("_answered"):
        await update.message.reply_text(
            "Use the buttons above — <b>💡 Explain More</b> or <b>➡️ Next Question</b>.",
            parse_mode=ParseMode.HTML,
        )
        return

    user_text = update.message.text.strip()
    await _handle_answer(db, user.id, update.effective_chat.id, user_text, context)


# ── Voice message handler ──────────────────────────────────────────────────────

async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db: Database = context.bot_data["db"]
    user = update.effective_user
    db.upsert_user(user.id, user.username or "", user.first_name or "")

    pending = db.get_pending_question(user.id)
    if not pending:
        await _send_question(db, user.id, update.effective_chat.id, context)
        return

    if pending.get("_answered"):
        await update.message.reply_text(
            "Use the buttons above — <b>💡 Explain More</b> or <b>➡️ Next Question</b>.",
            parse_mode=ParseMode.HTML,
        )
        return

    # Download the voice file from Telegram
    voice = update.message.voice
    try:
        file = await context.bot.get_file(voice.file_id)
        audio_bytes = await file.download_as_bytearray()
    except Exception as exc:
        logger.error("Failed to download voice file: %s", exc)
        await update.message.reply_text("⚠️ Couldn't download your voice message. Please try typing your answer.")
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        result = await evaluate_voice_answer(pending, bytes(audio_bytes), mime_type="audio/ogg")
    except GeminiQuotaExceeded:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="📵 <b>Daily AI quota reached.</b> The free tier resets at midnight UTC. Try again then!",
            parse_mode=ParseMode.HTML,
        )
        return
    except Exception as exc:
        logger.error("evaluate_voice_answer failed: %s", exc)
        await update.message.reply_text("⚠️ Couldn't process your voice message. Please try typing your answer.")
        return

    transcription = result.get("transcription", "")
    if transcription:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"🎤 <i>You said:</i> \"{_esc(transcription)}\"",
            parse_mode=ParseMode.HTML,
        )

    # Re-use _handle_answer logic with the transcription as the user_answer,
    # but the evaluation is already done — inject result directly.
    is_correct = result.get("is_correct", False)
    score      = result.get("score", 0.0)
    feedback   = result.get("feedback", "")
    correction = result.get("correction", "")

    if score >= 0.9:
        header = "✅ <b>Correct!</b>"
    elif score >= 0.5:
        header = "🟡 <b>Partially correct</b>"
    else:
        header = "❌ <b>Not quite</b>"

    parts = [header]
    if feedback:
        parts.append(_esc(feedback))
    if correction:
        parts.append(f"<b>Correction:</b> {_esc(correction)}")
    parts.append(f"<b>Correct answer:</b> <code>{_esc(pending.get('correct_answer', ''))}</code>")
    if pending.get("explanation"):
        parts.append(f"💡 <i>{_esc(pending['explanation'])}</i>")

    old_perf = db.get_topic_performance(user.id, pending["topic"], pending["subtopic"])
    from adaptive import adjust_difficulty, next_review_interval
    new_diff = adjust_difficulty(old_perf.get("difficulty", 2.0), is_correct)
    new_ri   = next_review_interval(old_perf.get("review_interval", 1.0), is_correct)
    db.update_performance(user.id, pending["topic"], pending["subtopic"], is_correct, score, new_diff, new_ri)

    is_exam = pending.get("_exam_mode", False)
    if not is_exam:
        streak, is_new_day = db.update_streak(user.id)
        if is_new_day:
            milestone_msg = _STREAK_MILESTONES.get(streak)
            if milestone_msg:
                parts.append(milestone_msg)
            elif streak > 1:
                parts.append(f"🔥 <b>{streak}-day streak!</b>")

    msg_id = pending.get("_question_msg_id")
    if msg_id:
        try:
            await context.bot.edit_message_reply_markup(
                chat_id=update.effective_chat.id, message_id=msg_id, reply_markup=None
            )
        except Exception:
            pass

    if is_exam:
        keyboard = _exam_keyboard()
        exam_state = db.get_exam_state(user.id)
        if exam_state:
            exam_state["done"] += 1
            exam_state["results"].append({
                "topic":      pending["topic"],
                "subtopic":   pending["subtopic"],
                "is_correct": is_correct,
                "score":      score,
            })
            db.set_exam_state(user.id, exam_state)
    else:
        keyboard = _action_keyboard()

    pending["_answered"]      = True
    pending["_user_answer"]   = transcription
    pending["_last_feedback"] = feedback + (f"\nCorrection: {correction}" if correction else "")
    pending["_explain_depth"] = 0
    db.set_pending_question(user.id, pending)

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="\n\n".join(parts),
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )


# ── Callback handler (inline keyboard answers) ────────────────────────────────

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    db: Database = context.bot_data["db"]
    user = query.from_user
    db.upsert_user(user.id, user.username or "", user.first_name or "")

    data = query.data

    if data.startswith("answer:"):
        user_answer = data.split(":", 1)[1]
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        await _handle_answer(db, user.id, query.message.chat_id, user_answer, context)

    elif data == "next_q":
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        db.set_pending_question(user.id, None)
        await _send_question(db, user.id, query.message.chat_id, context, is_first=True)

    elif data == "exam_next":
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        exam_state = db.get_exam_state(user.id)
        if not exam_state or not exam_state.get("active"):
            # Exam somehow lost state — fall back to normal practice
            db.set_pending_question(user.id, None)
            await _send_question(db, user.id, query.message.chat_id, context, is_first=True)
            return
        db.set_pending_question(user.id, None)
        if exam_state["done"] >= exam_state["total"]:
            await _send_exam_summary(db, user.id, query.message.chat_id, context, exam_state)
        else:
            await _send_exam_question(db, user.id, query.message.chat_id, context, exam_state)

    elif data == "hint":
        pending = db.get_pending_question(user.id)
        if not pending or pending.get("_answered") or pending.get("_exam_mode"):
            return
        hint_count = pending.get("_hint_count", 0)
        if hint_count >= _MAX_HINTS:
            return
        # Use pre-bundled hints from the question JSON first (no Gemini call needed).
        # Fall back to a live Gemini call for backward compat with older pending questions.
        bundled_key = f"hint_{hint_count + 1}"
        hint_text   = pending.get(bundled_key)
        if not hint_text:
            try:
                await context.bot.send_chat_action(chat_id=query.message.chat_id, action="typing")
                hint_text = await get_hint(pending, hint_count)
            except GeminiQuotaExceeded:
                await context.bot.send_message(
                    chat_id=query.message.chat_id,
                    text="📵 <b>Daily AI quota reached.</b> The free tier resets at midnight UTC.",
                    parse_mode=ParseMode.HTML,
                )
                return
            except Exception as exc:
                logger.error("get_hint failed: %s", exc)
                return
        pending["_hint_count"] = hint_count + 1
        db.set_pending_question(user.id, pending)
        # Update the question keyboard: advance hint count (may remove button if at max)
        new_kb = _question_keyboard(pending, hint_count + 1)
        try:
            await query.edit_message_reply_markup(reply_markup=new_kb)
        except Exception:
            pass
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"🔍 <b>Hint:</b> {_esc(hint_text)}",
            parse_mode=ParseMode.HTML,
        )

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
        pending["_last_feedback"] = explanation[:500]
        db.set_pending_question(user.id, pending)
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=_esc(explanation),
            parse_mode=ParseMode.HTML,
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
    app.add_handler(CommandHandler("exam",   cmd_exam))
    app.add_handler(CommandHandler("stats",  cmd_stats))
    app.add_handler(CommandHandler("topic",  cmd_topic))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    logger.info("Bot is running…")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
