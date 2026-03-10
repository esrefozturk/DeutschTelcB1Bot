"""
Adaptive learning engine for the TELC B1 bot.

Responsibilities:
  - Define the B1 topic/subtopic taxonomy
  - Pick the next (topic, subtopic, question_type, difficulty) tuple
    weighted by the user's past performance + SRS overdue factor
  - Adjust difficulty and review interval after each answer
"""

import random
from datetime import datetime, timezone
from typing import Tuple, List, Dict

# ── Topic taxonomy ────────────────────────────────────────────────────────────

TOPICS: Dict[str, Dict] = {
    "grammar": {
        "label": "Grammar",
        "subtopics": {
            "cases":              "Nominativ / Akkusativ / Dativ / Genitiv",
            "verb_conjugation":   "Present, Perfekt, Präteritum, Future",
            "modal_verbs":        "können, müssen, dürfen, wollen, sollen, möchten",
            "prepositions":       "two-way, dative-only, accusative-only prepositions",
            "relative_clauses":   "Relativsätze with der/die/das/dessen",
            "konjunktiv_2":       "würde, hätte, wäre – polite requests and hypotheticals",
            "passive":            "Passiv with werden / sein",
            "adjective_endings":  "strong, weak, mixed declension",
            "conjunctions":       "weil, dass, obwohl, wenn, als, damit, um…zu",
        },
    },
    "vocabulary": {
        "label": "Vocabulary",
        "subtopics": {
            "daily_life":         "Everyday activities and time expressions",
            "work_profession":    "Job types, workplace, applications, CVs",
            "travel_transport":   "Transportation, directions, booking, accommodation",
            "health_body":        "Body parts, illnesses, doctor/pharmacy visits",
            "family_relationships": "Family, personal relationships, social life",
            "education":          "School, university, courses, qualifications",
            "shopping_money":     "Prices, banking, consumer vocabulary",
            "housing":            "Rooms, furniture, renting, utilities",
            "food_cooking":       "Ingredients, recipes, restaurant, ordering",
            "culture_leisure":    "Hobbies, sports, events, media",
            "environment":        "Nature, recycling, climate, sustainability",
        },
    },
    "reading": {
        "label": "Reading Comprehension",
        "subtopics": {
            "main_idea":          "Identifying the main idea of a text",
            "find_information":   "Locating specific facts in notices, ads, articles",
            "text_types":         "Emails, letters, instructions, news articles",
            "inference":          "Reading between the lines – implied meaning",
        },
    },
    "writing": {
        "label": "Writing",
        "subtopics": {
            "formal_letter":      "Formal letters – applications, complaints, requests",
            "informal_message":   "Casual emails/messages to friends or colleagues",
            "opinion_text":       "Expressing and justifying personal opinions",
            "describing_events":  "Describing past events or experiences",
        },
    },
}

# Which question types are available (and valid) for each top-level topic
QUESTION_TYPES_BY_TOPIC: Dict[str, List[str]] = {
    "grammar": [
        "fill_blank",
        "multiple_choice",
        "error_correction",
        "sentence_building",
        "word_bank_fill",
    ],
    "vocabulary": [
        "multiple_choice",
        "translation_to_german",
        "translation_to_english",
        "fill_blank",
        "sentence_building",
        "word_bank_fill",
    ],
    "reading": [
        "multiple_choice",
        "short_answer",
        "true_false",
        "text_heading_matching",
        "situation_matching",
    ],
    "writing": [
        "short_answer",
        "sentence_building",
        "translation_to_german",
    ],
}

# ── Difficulty helpers ────────────────────────────────────────────────────────

MIN_DIFFICULTY = 1.0
MAX_DIFFICULTY = 5.0
STEP_UP        = 0.5   # after a correct answer
STEP_DOWN      = 0.75  # after a wrong answer  (bigger step down → protective)


def adjust_difficulty(current: float, is_correct: bool) -> float:
    if is_correct:
        return min(MAX_DIFFICULTY, current + STEP_UP)
    else:
        return max(MIN_DIFFICULTY, current - STEP_DOWN)


def difficulty_label(d: float) -> str:
    if d <= 1.5:
        return "beginner"
    if d <= 2.5:
        return "elementary"
    if d <= 3.5:
        return "intermediate"
    if d <= 4.5:
        return "upper-intermediate"
    return "advanced"


# ── Spaced Repetition (SRS) ───────────────────────────────────────────────────

SRS_MIN_INTERVAL = 1.0   # days
SRS_MAX_INTERVAL = 30.0  # days


def next_review_interval(current_interval: float, is_correct: bool) -> float:
    """
    Compute the next SRS review interval in days.
      Correct → double the interval (capped at 30 days)
      Wrong   → reset to 1 day
    """
    if is_correct:
        return min(SRS_MAX_INTERVAL, current_interval * 2.0)
    return SRS_MIN_INTERVAL


def _srs_factor(last_tested: str, review_interval: float) -> float:
    """
    Return a weight multiplier based on how overdue a subtopic is for review.
    Subtopics never seen get a high priority factor of 2.0.
    Subtopics that are on schedule get 1.0.
    Each day overdue adds 0.3 (capped at 14 overdue days → max factor ~5.2).
    """
    if not last_tested:
        return 2.0  # never tested → high priority
    try:
        last_dt = datetime.fromisoformat(last_tested)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        days_since = (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400
        overdue    = max(0.0, days_since - review_interval)
        return 1.0 + min(overdue, 14.0) * 0.3
    except Exception:
        return 1.0


# ── Topic selection ───────────────────────────────────────────────────────────

# Base topic weights (tune as needed)
_BASE_WEIGHTS: Dict[str, float] = {
    "grammar":    3.0,
    "vocabulary": 3.0,
    "reading":    2.0,
    "writing":    1.5,
}


def _error_rate(correct: int, incorrect: int) -> float:
    total = correct + incorrect
    return incorrect / total if total >= 3 else 0.5  # default 50% for unseen


def pick_next_params(
    performance_rows: list[Dict],
) -> Tuple[str, str, str, float]:
    """
    Return (topic, subtopic, question_type, difficulty) for the next question.

    Selection logic:
      weight = base_topic_weight × (1 + error_rate²) × srs_factor
      srs_factor rises the longer a subtopic is overdue for review,
      so items due for spaced repetition surface more often.
    """
    perf: Dict[Tuple[str, str], Dict] = {
        (r["topic"], r["subtopic"]): r for r in performance_rows
    }

    candidates: List[Tuple[float, str, str, float]] = []

    for topic, topic_data in TOPICS.items():
        base_w = _BASE_WEIGHTS.get(topic, 1.0)
        for subtopic in topic_data["subtopics"]:
            key = (topic, subtopic)
            if key in perf:
                p    = perf[key]
                er   = _error_rate(p["correct"], p["incorrect"])
                diff = p["difficulty"]
                srs  = _srs_factor(p.get("last_tested", ""), p.get("review_interval", 1.0))
            else:
                er   = 0.5   # unseen → treat as 50% error (needs practice)
                diff = 2.0
                srs  = 2.0   # never tested → high priority

            weight = base_w * (1.0 + er ** 2) * srs
            candidates.append((weight, topic, subtopic, diff))

    # Weighted random selection
    total_weight = sum(c[0] for c in candidates)
    r = random.uniform(0, total_weight)
    cumulative = 0.0
    chosen = candidates[-1]  # fallback
    for c in candidates:
        cumulative += c[0]
        if r <= cumulative:
            chosen = c
            break

    _, topic, subtopic, difficulty = chosen
    question_type = random.choice(QUESTION_TYPES_BY_TOPIC[topic])
    return topic, subtopic, question_type, difficulty


def get_subtopic_description(topic: str, subtopic: str) -> str:
    return TOPICS[topic]["subtopics"].get(subtopic, subtopic)


def all_subtopics_flat() -> List[Tuple[str, str]]:
    result = []
    for topic, data in TOPICS.items():
        for subtopic in data["subtopics"]:
            result.append((topic, subtopic))
    return result
