"""
Gemini API client (uses the current google-genai SDK).

Two public coroutines:
  generate_question(topic, subtopic, question_type, difficulty, context) -> dict
  evaluate_answer(question_data, user_answer)                             -> dict
"""

import json
import re
import asyncio
from typing import Dict, Optional

from google import genai
from google.genai import types

from adaptive import get_subtopic_description, difficulty_label

# Module-level client — initialised once by init_gemini()
_client: Optional[genai.Client] = None
MODEL = "gemini-2.5-flash"   # free tier on this project: 5 RPM / 20 RPD


def init_gemini(api_key: str):
    global _client
    # Use default v1beta endpoint — all gemini-2.x models live there
    _client = genai.Client(api_key=api_key)


def _client_or_raise() -> genai.Client:
    if _client is None:
        raise RuntimeError("Call init_gemini(api_key) before using gemini_client.")
    return _client


# ── Prompt builders ───────────────────────────────────────────────────────────

_QUESTION_SCHEMA = """
{
  "question":      "<the question text, in German or English as appropriate>",
  "question_type": "<one of: multiple_choice | fill_blank | translation_to_german | translation_to_english | error_correction | sentence_building | short_answer>",
  "options":       ["A) ...", "B) ...", "C) ...", "D) ..."],
  "correct_answer": "<the exact correct answer or the letter A/B/C/D for multiple choice>",
  "explanation":   "<1-2 sentences explaining the rule or meaning in English>",
  "topic":         "<topic string>",
  "subtopic":      "<subtopic string>",
  "difficulty":    <number 1-5>
}
""".strip()


def _build_question_prompt(
    topic: str,
    subtopic: str,
    question_type: str,
    difficulty: float,
    context: Optional[str],
) -> str:
    diff_label    = difficulty_label(difficulty)
    subtopic_desc = get_subtopic_description(topic, subtopic)

    type_instructions = {
        "multiple_choice": (
            "Create a multiple-choice question with exactly 4 options (A, B, C, D). "
            "Only one option is correct. Make the distractors plausible."
        ),
        "fill_blank": (
            "Create a sentence with ONE blank marked as ___. "
            "The correct answer fills the blank. Do NOT include options."
        ),
        "translation_to_german": (
            "Give an English sentence or phrase that the learner must translate into German. "
            "The correct_answer field contains the German translation."
        ),
        "translation_to_english": (
            "Give a German sentence that the learner must translate into English. "
            "The correct_answer field contains the English translation."
        ),
        "error_correction": (
            "Write a German sentence that contains exactly ONE grammatical error. "
            "The learner must identify and correct it. "
            "The correct_answer is the fully corrected sentence."
        ),
        "sentence_building": (
            "Provide a scrambled set of German words (comma-separated) that the learner "
            "must arrange into a correct sentence. "
            "The correct_answer is the properly ordered sentence."
        ),
        "short_answer": (
            "Ask an open-ended question about a short German text or scenario. "
            "The correct_answer should be a model answer (1-2 sentences)."
        ),
    }

    instruction   = type_instructions.get(question_type, "Create a question.")
    context_block = f"\n\nAvoid repeating these recently asked topics: {context}" if context else ""

    return f"""You are an expert German language teacher preparing a student for the TELC B1 exam.

Task: Generate ONE question.
  • Topic: {topic} — {subtopic} ({subtopic_desc})
  • Question type: {question_type}
  • Difficulty level: {difficulty:.1f}/5 ({diff_label})
  • Target exam: TELC B1 (Common European Framework B1){context_block}

Instructions for this question type:
{instruction}

Important rules:
- Keep the question realistic and relevant to everyday B1 German situations.
- At difficulty 1-2: use simple vocabulary and common structures.
- At difficulty 3: use standard B1 vocabulary and grammar.
- At difficulty 4-5: use less common vocabulary, complex sentence structures.
- The explanation must be in ENGLISH and genuinely teach the rule.
- Return ONLY the JSON object below — no markdown fences, no extra text.

Required JSON schema:
{_QUESTION_SCHEMA}
"""


_EVAL_SCHEMA = """
{
  "is_correct":  <true | false>,
  "score":       <0.0 – 1.0>,
  "feedback":    "<encouraging, constructive feedback in English — 1-3 sentences>",
  "correction":  "<corrected answer with brief explanation, only include if is_correct is false>"
}
""".strip()


def _build_eval_prompt(question_data: Dict, user_answer: str) -> str:
    q_type = question_data.get("question_type", "")
    options_block = ""
    if q_type == "multiple_choice" and "options" in question_data:
        options_block = "\nOptions were:\n" + "\n".join(question_data["options"])

    leniency = (
        "Be strict about the grammar concept being tested but lenient on minor spelling "
        "variations. For translations accept paraphrases if the meaning is correct. "
        "For multiple-choice, only accept the exact letter (A/B/C/D) or the full text of the option."
    )
    if q_type in ("translation_to_german", "translation_to_english", "short_answer"):
        leniency = (
            "Accept any answer that conveys the correct meaning, even if wording differs slightly. "
            "Be encouraging and highlight what was good even in partial answers."
        )

    return f"""You are a German language teacher evaluating a TELC B1 learner's answer.

Question ({q_type}):
{question_data.get('question', '')}
{options_block}

Expected correct answer: {question_data.get('correct_answer', '')}

Learner's answer: {user_answer}

Evaluation rules:
- {leniency}
- score: 1.0 = fully correct, 0.5 = partially correct (right idea, wrong form), 0.0 = wrong.
- is_correct should be true if score >= 0.7.
- feedback must be warm and pedagogically useful.
- correction: include ONLY if is_correct is false; show the correct German with English explanation.

Return ONLY the JSON object below — no markdown fences, no extra text.

Required JSON schema:
{_EVAL_SCHEMA}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Dict:
    """Strip markdown fences if the model adds them despite instructions."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _gen_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        temperature=0.9,
        response_mime_type="application/json",  # v1beta supports JSON mode
    )


# ── Public API ────────────────────────────────────────────────────────────────

async def generate_question(
    topic: str,
    subtopic: str,
    question_type: str,
    difficulty: float,
    recent_context: Optional[str] = None,
) -> Dict:
    prompt  = _build_question_prompt(topic, subtopic, question_type, difficulty, recent_context)
    client  = _client_or_raise()
    loop    = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=_gen_config(),
        ),
    )
    data = _extract_json(response.text)
    data.setdefault("topic",      topic)
    data.setdefault("subtopic",   subtopic)
    data.setdefault("difficulty", difficulty)
    return data


async def evaluate_answer(question_data: Dict, user_answer: str) -> Dict:
    prompt  = _build_eval_prompt(question_data, user_answer)
    client  = _client_or_raise()
    loop    = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=_gen_config(),
        ),
    )
    data = _extract_json(response.text)
    data.setdefault("is_correct", False)
    data.setdefault("score",      0.0)
    data.setdefault("feedback",   "")
    return data
