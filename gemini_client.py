"""
Gemini API client (uses the current google-genai SDK).

Two public coroutines:
  generate_question(topic, subtopic, question_type, difficulty, context) -> dict
  evaluate_answer(question_data, user_answer)                             -> dict

Model fallback chain (paid tier-1, only models available on this project):
  1. gemini-2.5-pro       —   150 RPM /   1 000 RPD (best quality primary)
  2. gemini-2.5-flash     — 1 000 RPM /  10 000 RPD (fallback)
  3. gemini-2.5-flash-lite — 4 000 RPM / unlimited RPD (last resort)

If a model returns 429 (quota exhausted), the next one is tried automatically.
"""

import json
import logging
import re
import asyncio
from typing import Dict, List, Optional

from google import genai
from google.genai import types

from adaptive import get_subtopic_description, difficulty_label

logger = logging.getLogger(__name__)

# Module-level client — initialised once by init_gemini()
_client: Optional[genai.Client] = None

# Primary: pro (best quality); fallbacks: flash → flash-lite.
MODELS: List[str] = [
    "gemini-2.5-pro",        #   150 RPM /  1 000 RPD — best quality primary
    "gemini-2.5-flash",      # 1 000 RPM / 10 000 RPD — fallback
    "gemini-2.5-flash-lite", # 4 000 RPM / unlimited  — last resort
]


def init_gemini(api_key: str):
    global _client
    _client = genai.Client(api_key=api_key)


def _client_or_raise() -> genai.Client:
    if _client is None:
        raise RuntimeError("Call init_gemini(api_key) before using gemini_client.")
    return _client


# ── Prompt builders ───────────────────────────────────────────────────────────

_QUESTION_SCHEMA = """
{
  "question":         "<the question text, in German or English as appropriate>",
  "question_type":    "<one of: multiple_choice | fill_blank | translation_to_german | translation_to_english | error_correction | sentence_building | short_answer>",
  "options":          ["A) ...", "B) ...", "C) ...", "D) ..."],
  "correct_answer":   "<the exact correct answer or the letter A/B/C/D for multiple choice>",
  "error_introduced": "<error_correction only: describe the exact error you put in the sentence, e.g. 'wrong adjective ending: interessante → interessanten'; empty string for all other types>",
  "explanation":      "<1-2 sentences explaining the rule or meaning in English>",
  "hint_1":           "<a gentle hint that nudges the learner without revealing the answer>",
  "hint_2":           "<a stronger follow-up hint that is more specific but still does NOT reveal the answer>",
  "topic":            "<topic string>",
  "subtopic":         "<subtopic string>",
  "difficulty":       <number 1-5>
}
""".strip()


def _build_question_prompt(
    topic: str,
    subtopic: str,
    question_type: str,
    difficulty: float,
    context: Optional[str],
    avoided_questions: Optional[List[str]] = None,
) -> str:
    diff_label    = difficulty_label(difficulty)
    subtopic_desc = get_subtopic_description(topic, subtopic)

    type_instructions = {
        "multiple_choice": (
            "Create a multiple-choice question with exactly 4 options (A, B, C, D). "
            "Only one option is correct. Make the distractors plausible. "
            "If the question involves choosing a word or phrase that belongs in a sentence, "
            "mark the gap in the sentence with ___ so the learner knows where it goes."
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
            "Write a German sentence that contains exactly ONE deliberate grammatical error. "
            "The learner must identify and correct it. "
            "IMPORTANT: A grammatical error means a FORM violation — wrong inflection, wrong "
            "conjugation, wrong auxiliary, wrong word order — NOT a semantic or word-choice issue. "
            "IMPORTANT: Before writing the sentence, decide on a specific grammatical error "
            "you will introduce (e.g. wrong case ending, wrong verb form, wrong word order). "
            "Then write a sentence with that error clearly present. "
            "Fill 'error_introduced' with a short description of the exact error you introduced "
            "(e.g. 'Perfekt formed with 'haben' instead of 'sein' for a motion verb'). "
            "The correct_answer is the full corrected sentence — never a single letter. "
            "Set 'options' to an empty array []. "
            "Only change the minimum necessary to fix the error — do not add words, "
            "restructure, or rephrase the sentence beyond what the fix requires. "
            "The corrected sentence must have EXACTLY the same number of words as the erroneous sentence. "
            "Double-check: the question sentence and correct_answer must differ by exactly the one error. "
            "CRITICAL — the error must be UNAMBIGUOUSLY wrong: a native speaker would immediately "
            "flag it as a grammatical mistake with no room for debate. "
            "Good error types: wrong case ending (e.g. 'mit dem Mann' → 'mit der Mann'), "
            "wrong Perfekt auxiliary (haben/sein), wrong past participle form, "
            "wrong adjective ending, verb not in second position in a main clause, "
            "wrong separable prefix placement, wrong reflexive pronoun. "
            "FORBIDDEN error types (do NOT use these — they are ambiguous or stylistically valid): "
            "modal verb substitution where both verbs are grammatically valid in context "
            "(e.g. replacing 'muss' with 'darf' or 'soll' — these change meaning, not grammar), "
            "mixing Indikativ and Konjunktiv II when both moods are contextually defensible, "
            "'gerne' vs 'gern' (both correct), word order variations that are regionally accepted, "
            "any change where a native speaker might say 'both are fine', "
            "any change that makes the sentence semantically odd rather than grammatically wrong. "
            "CRITICAL — the 'question' field must contain ONLY the erroneous German sentence itself. "
            "Do NOT include any instruction text, preamble, or phrases like "
            "'Korrigieren Sie den Fehler im folgenden Satz:' — just the raw sentence."
        ),
        "sentence_building": (
            "Provide a scrambled set of German words (comma-separated) that the learner "
            "must arrange into a correct sentence. "
            "The correct_answer is the properly ordered sentence. "
            "IMPORTANT: each token in the scramble must appear exactly as it will in the "
            "final sentence — do NOT split compound nouns (e.g. provide 'Kunstausstellungen' "
            "as one token, never as 'Kunst' and 'Ausstellungen' separately). "
            "The learner should only need to reorder words, not merge or split them. "
            "CRITICAL — the 'question' field must contain ONLY the comma-separated word list. "
            "Do NOT include any instruction text, preamble, or phrases like "
            "'Ordnen Sie die folgenden Wörter zu einem korrekten Satz:' — just the raw word list."
        ),
        "short_answer": (
            "Ask an open-ended question about a short German text or scenario. "
            "The correct_answer should be a model answer (1-2 sentences)."
        ),
    }

    instruction   = type_instructions.get(question_type, "Create a question.")

    # Reading comprehension questions MUST embed the source text.
    # Without it the learner has nothing to read — they cannot answer.
    if topic == "reading":
        passage_types = {
            "find_information": "a short notice, advertisement, or announcement (Anzeige/Bekanntmachung)",
            "main_idea":        "a short newspaper article or blog post excerpt",
            "inference":        "a short letter, email, or narrative passage",
            "text_types":       "a short sample of the relevant text type (email, letter, instructions, or news article)",
        }
        passage_hint = passage_types.get(subtopic, "a short German text (4-8 sentences)")
        instruction += (
            f"\n\nCRITICAL — Reading comprehension requirement: "
            f"The 'question' field MUST begin with the full German passage ({passage_hint}), "
            f"followed by a blank line, followed by the comprehension question in German. "
            f"Format exactly like this:\n"
            f"  <German passage text here — 4 to 8 sentences>\n\n"
            f"  <The actual question about the passage>\n\n"
            f"Do NOT ask about a text that you have not included in the question field. "
            f"The learner can only see the 'question' field — they have no other source."
        )

    context_block = f"\n\nAvoid repeating these recently asked topics: {context}" if context else ""

    avoided_block = ""
    if avoided_questions:
        avoided_list = "\n".join(f"  - {q}" for q in avoided_questions[:25])
        avoided_block = f"\n\nDo NOT create a question identical or very similar to any of these recently asked questions:\n{avoided_list}"

    return f"""You are an expert German language teacher preparing a student for the TELC B1 exam.

Task: Generate ONE question.
  • Topic: {topic} — {subtopic} ({subtopic_desc})
  • Question type: {question_type}
  • Difficulty level: {difficulty:.1f}/5 ({diff_label})
  • Target exam: TELC B1 (Common European Framework B1){context_block}{avoided_block}

Instructions for this question type:
{instruction}

Important rules:
- Keep the question realistic and relevant to everyday B1 German situations.
- At difficulty 1-2: use simple vocabulary and common structures.
- At difficulty 3: use standard B1 vocabulary and grammar.
- At difficulty 4-5: use less common vocabulary, complex sentence structures.
- The explanation must be in ENGLISH and genuinely teach the rule.
- hint_1 must gently nudge without revealing the answer; hint_2 can be more specific but must NOT state the answer.
- Return ONLY the JSON object below — no markdown fences, no extra text.

Required JSON schema:
{_QUESTION_SCHEMA}
"""


_EVAL_SCHEMA = """
{
  "is_correct":  <true | false>,
  "score":       <0.0 – 1.0>,
  "feedback":    "<encouraging, constructive feedback in English — 1-3 sentences>",
  "correction":  "<PLAIN STRING — corrected answer with brief explanation in one string, only include if is_correct is false. Do NOT nest objects here.>"
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
    elif q_type == "error_correction":
        leniency = (
            "Accept any answer that correctly fixes the same grammatical error, "
            "even if the exact wording differs from the model answer. "
            "The key is whether the learner identified and fixed the right grammatical "
            "issue — not whether their corrected sentence matches word-for-word."
        )

    error_introduced_block = ""
    if q_type == "error_correction" and question_data.get("error_introduced"):
        error_introduced_block = f"\nThe intentional error in the question was: {question_data['error_introduced']}"

    return f"""You are a German language teacher evaluating a TELC B1 learner's answer.

Question ({q_type}):
{question_data.get('question', '')}
{options_block}

Expected correct answer: {question_data.get('correct_answer', '')}{error_introduced_block}

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


def _text_gen_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(temperature=0.7)


class GeminiQuotaExceeded(Exception):
    """Raised when all models in the fallback chain have hit their daily quota."""


def _is_quota_error(exc: Exception) -> bool:
    """Return True only for genuine quota-exhaustion errors (HTTP 429).

    503 UNAVAILABLE is a transient server error, not a quota error — treating
    it as quota exhaustion causes the bot to tell users their daily limit is
    reached and to burn through the model fallback chain on a simple blip.
    503s are allowed to bubble up so the caller can show a generic retry message.
    """
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()


async def _call_with_fallback(prompt: str, config: Optional[types.GenerateContentConfig] = None) -> str:
    """Try each model in MODELS order; fall back on 429 quota errors."""
    if config is None:
        config = _gen_config()
    client = _client_or_raise()
    last_exc: Optional[Exception] = None
    for model in MODELS:
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda m=model: client.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=config,
                ),
            )
            if model != MODELS[0]:
                logger.info("Fell back to model: %s", model)
            return response.text
        except Exception as exc:
            if _is_quota_error(exc):
                logger.warning("Quota exhausted on %s, trying next model.", model)
                last_exc = exc
                continue
            raise  # non-quota errors bubble up immediately
    raise GeminiQuotaExceeded("All models exhausted their daily quota.") from last_exc


# ── Validators ────────────────────────────────────────────────────────────────

async def _verify_error_correction(q: str, ans: str, error_introduced: str) -> None:
    """Second-opinion check: confirm the question sentence has a real grammar error.

    Prompts Gemini to distinguish genuine form violations (wrong case ending,
    wrong auxiliary, wrong conjugation, wrong word order) from semantic swaps
    (e.g. müssen → dürfen) that are grammatically valid in context.
    Raises ValueError if the sentence appears already correct, triggering a retry.
    """
    prompt = (
        "You are a strict German grammar checker.\n\n"
        f'Does this German sentence contain a real grammatical error?\n\nSentence: "{q}"\n'
        f'Claimed error: "{error_introduced}"\n'
        f'Supposedly corrected to: "{ans}"\n\n'
        "A REAL grammatical error is a FORM violation:\n"
        "- Wrong case ending (Nominativ/Akkusativ/Dativ/Genitiv)\n"
        "- Wrong verb conjugation (person, number, tense)\n"
        "- Wrong Perfekt auxiliary (haben vs sein)\n"
        "- Wrong word order (e.g. verb not in V2 position)\n"
        "- Wrong separable prefix placement\n"
        "- Wrong adjective ending\n\n"
        "NOT a real grammatical error:\n"
        "- Swapping one modal verb for another when both are grammatically valid "
        "(e.g. 'muss' → 'darf' — both are correct modal verb forms; only meaning differs)\n"
        "- Any word substitution where both words are correctly inflected in context\n"
        "- Stylistic or register differences\n\n"
        'Return ONLY this JSON: {"has_real_error": true/false, "reason": "<one sentence>"}'
    )
    try:
        text   = await _call_with_fallback(prompt, config=_gen_config())
        result = _extract_json(text)
        if not result.get("has_real_error", True):
            raise ValueError(
                f"error_correction sentence has no real grammatical error "
                f"(only a semantic/meaning difference): {q!r} — "
                f"validator: {result.get('reason', '')}"
            )
    except ValueError:
        raise
    except Exception as exc:
        # Parsing failure or network issue — don't block question generation.
        logger.warning("_verify_error_correction check failed (letting through): %s", exc)


# ── Public API ────────────────────────────────────────────────────────────────

async def generate_question(
    topic: str,
    subtopic: str,
    question_type: str,
    difficulty: float,
    recent_context: Optional[str] = None,
    avoided_questions: Optional[List[str]] = None,
) -> Dict:
    prompt = _build_question_prompt(topic, subtopic, question_type, difficulty, recent_context, avoided_questions)
    text   = await _call_with_fallback(prompt)
    data   = _extract_json(text)
    # Always force topic/subtopic to our chosen values — Gemini sometimes echoes
    # back augmented strings like "verb_conjugation (Perfekt)" which break the
    # TOPICS dict lookup in bot.py.
    data["topic"]    = topic
    data["subtopic"] = subtopic
    data.setdefault("difficulty", difficulty)

    # Fix: Gemini sometimes prepends a German instruction phrase to the question
    # field for sentence_building ("Ordnen Sie die folgenden Wörter zu einem
    # korrekten Satz: ...") and error_correction ("Korrigieren Sie den Fehler im
    # folgenden Satz: ...").  Strip it so validation and display are not broken.
    q_type = data.get("question_type", "")
    if q_type in ("sentence_building", "error_correction"):
        q_text = data.get("question", "")
        # Match a German instruction prefix of 5-100 chars ending with ": "
        stripped = re.sub(r'^[A-ZÄÖÜ][^:]{4,99}:\s+', '', q_text)
        if stripped and stripped != q_text:
            logger.info("Stripped instruction prefix from %s question", q_type)
            data["question"] = stripped

    # Fix: for non-MC question types, Gemini occasionally generates options AND
    # sets correct_answer to a letter ("A"/"B"/…) instead of the actual answer
    # text.  Resolve the letter to the full option text when this happens.
    if (
        data.get("question_type") != "multiple_choice"
        and isinstance(data.get("correct_answer"), str)
        and data["correct_answer"].strip().upper() in ("A", "B", "C", "D")
        and data.get("options")
    ):
        letter = data["correct_answer"].strip().upper()
        idx    = ord(letter) - ord("A")
        opts   = data["options"]
        if 0 <= idx < len(opts):
            opt = opts[idx]
            if len(opt) > 2 and opt[1] in ") .":  # strip "A) " / "A. " prefix
                opt = opt[2:].strip()
            data["correct_answer"] = opt

    # Validate sentence_building questions: every scrambled token must appear
    # verbatim (case-insensitive) in the correct answer.  If a token is missing
    # the model likely split a compound noun — discard and retry.
    if data.get("question_type") == "sentence_building":
        answer_lower = data.get("correct_answer", "").lower()
        tokens = [t.strip() for t in data.get("question", "").split(",") if t.strip()]
        bad = [t for t in tokens if t.lower() not in answer_lower]
        if bad:
            raise ValueError(
                f"sentence_building scramble contains tokens not found in correct_answer "
                f"(likely split compound noun): {bad!r}"
            )

    # Validate reading comprehension questions: the question field must contain
    # the passage text.  A short question field means the model forgot to include
    # the text — discard and retry rather than showing a passage-less question.
    if topic == "reading":
        q_text = data.get("question", "")
        # A passage + question should be at least ~150 chars; a bare question is ~60
        if len(q_text) < 150:
            raise ValueError(
                f"reading question missing passage text (question field too short: {len(q_text)} chars): {q_text!r}"
            )

    # Validate error_correction questions: if the question and correct_answer are
    # identical the model failed to introduce an actual error — discard silently.
    # Also reject if word counts differ: all valid error types (wrong case ending,
    # wrong auxiliary, wrong adjective ending, etc.) change one word's form, never
    # add or remove words.  A word-count mismatch means the model rewrote content
    # (e.g. "dessen Auto" → "den"), producing a semantically different sentence.
    if data.get("question_type") == "error_correction":
        q   = data.get("question", "").strip()
        ans = data.get("correct_answer", "").strip()
        if len(q.split()) != len(ans.split()):
            raise ValueError(
                f"error_correction question and correct_answer have different word counts "
                f"({len(q.split())} vs {len(ans.split())}), suggesting content was rewritten: {q!r} → {ans!r}"
            )
        if q == ans or not data.get("error_introduced", "").strip():
            raise ValueError(
                "error_correction question has no actual error "
                f"(question == correct_answer or error_introduced is empty): {q!r}"
            )
        # Second-opinion check: verify the sentence actually has a real grammar error,
        # not just a semantic/meaning substitution (e.g. muss → darf).
        await _verify_error_correction(q, ans, data.get("error_introduced", ""))

    return data


async def evaluate_answer(question_data: Dict, user_answer: str) -> Dict:
    prompt = _build_eval_prompt(question_data, user_answer)
    text   = await _call_with_fallback(prompt)
    data   = _extract_json(text)
    data.setdefault("is_correct", False)
    data.setdefault("score",      0.0)
    data.setdefault("feedback",   "")
    # Gemini occasionally returns `correction` as a nested dict instead of a
    # plain string.  Flatten it so the bot doesn't display a raw Python object.
    correction = data.get("correction")
    if isinstance(correction, dict):
        parts = []
        if correction.get("corrected_answer"):
            parts.append(correction["corrected_answer"])
        if correction.get("explanation"):
            parts.append(f"({correction['explanation']})")
        data["correction"] = " ".join(parts)
    return data


def _build_explain_prompt(
    question_data: Dict,
    user_answer: str,
    previous_feedback: str,
    depth: int,
) -> str:
    q_type = question_data.get("question_type", "")
    options_block = ""
    if q_type == "multiple_choice" and "options" in question_data:
        options_block = "\nOptions were:\n" + "\n".join(question_data["options"])

    follow_up = (
        "The learner wants a deeper explanation of this concept."
        if depth == 0
        else f"The learner is asking for even more clarification (follow-up #{depth + 1}). Go deeper or try a different angle."
    )

    return f"""You are a German language teacher helping a TELC B1 learner understand a concept deeply.

Question ({q_type}):
{question_data.get('question', '')}{options_block}

Correct answer: {question_data.get('correct_answer', '')}
Learner's answer: {user_answer}
Previous feedback given: {previous_feedback or 'None'}

{follow_up}

Please explain:
- WHY the correct answer is correct (the grammar rule or vocabulary meaning)
- WHY the learner's answer was wrong (if it was)
- The underlying German grammar rule with its name (e.g. "Dativ case", "Perfekt tense")
- A concrete memory tip or analogy to remember this rule
- 1-2 additional example sentences in German (with English translations)

Keep your explanation friendly, clear, and mostly in English. Bold German words/phrases using *asterisks*.
Format nicely with line breaks. Keep it to 4-6 short paragraphs maximum.
Do NOT return JSON — just plain prose with markdown formatting.
"""


def _build_hint_prompt(question_data: Dict, hint_count: int) -> str:
    q_type = question_data.get("question_type", "")
    options_block = ""
    if q_type == "multiple_choice" and "options" in question_data:
        options_block = "\nOptions:\n" + "\n".join(question_data["options"])

    depth = (
        "Give a gentle first hint — nudge the learner without revealing the answer."
        if hint_count == 0
        else "Give a stronger second hint — more specific, but still don't reveal the answer directly."
    )
    type_guidance = {
        "multiple_choice":        "Help eliminate one or two wrong options with a brief reason.",
        "fill_blank":             "Hint at the grammar rule or word form needed (e.g. 'Think about which case this preposition takes').",
        "translation_to_german":  "Hint at a key word or grammatical structure needed in German.",
        "translation_to_english": "Hint at the meaning of a key German word or phrase in the sentence.",
        "error_correction":       "Point to which part of the sentence contains the error (e.g. beginning/middle/end, or which word type is wrong).",
        "sentence_building":      "Suggest which word should come first, or name the grammatical rule that governs the word order.",
        "short_answer":           "Point to which part of the text or scenario contains the answer.",
    }.get(q_type, "Give a gentle directional hint.")

    return f"""You are helping a German B1 learner who is stuck on a question.

Question ({q_type}):
{question_data.get('question', '')}{options_block}

Correct answer (DO NOT reveal this): {question_data.get('correct_answer', '')}

{depth}
Type-specific guidance: {type_guidance}

Rules:
- Maximum 1-2 sentences
- Do NOT state the answer or any direct part of it
- Be warm and encouraging
- Write in English
"""


async def get_hint(question_data: Dict, hint_count: int = 0) -> str:
    prompt = _build_hint_prompt(question_data, hint_count)
    return (await _call_with_fallback(prompt, config=_text_gen_config())).strip()


async def explain_further(
    question_data: Dict,
    user_answer: str,
    previous_feedback: str,
    depth: int = 0,
) -> str:
    prompt = _build_explain_prompt(question_data, user_answer, previous_feedback, depth)
    text   = await _call_with_fallback(prompt, config=_text_gen_config())
    return text.strip()


# ── Voice evaluation ──────────────────────────────────────────────────────────

_VOICE_EVAL_SCHEMA = """
{
  "transcription": "<exact text of what the learner said>",
  "is_correct":    <true | false>,
  "score":         <0.0 – 1.0>,
  "feedback":      "<encouraging, constructive feedback in English — 1-3 sentences>",
  "correction":    "<corrected answer with brief explanation, only include if is_correct is false>"
}
""".strip()


def _build_voice_eval_prompt(question_data: Dict) -> str:
    q_type = question_data.get("question_type", "")
    options_block = ""
    if q_type == "multiple_choice" and "options" in question_data:
        options_block = "\nOptions were:\n" + "\n".join(question_data["options"])

    leniency = (
        "Be strict about the grammar concept being tested but lenient on minor pronunciation "
        "variations. For translations accept paraphrases if the meaning is correct. "
        "For multiple-choice, accept the letter (A/B/C/D) OR the spoken text of the option."
    )
    if q_type in ("translation_to_german", "translation_to_english", "short_answer"):
        leniency = (
            "Accept any spoken answer that conveys the correct meaning. "
            "Be encouraging and highlight what was good even in partial answers."
        )

    return f"""You are a German language teacher evaluating a TELC B1 learner's SPOKEN answer.

The audio attached is the learner's voice message response to this question.

Question ({q_type}):
{question_data.get('question', '')}
{options_block}

Expected correct answer: {question_data.get('correct_answer', '')}

Your tasks:
1. Transcribe exactly what the learner said in the audio.
2. Evaluate whether it answers the question correctly.

Evaluation rules:
- {leniency}
- score: 1.0 = fully correct, 0.5 = partially correct (right idea, wrong form), 0.0 = wrong.
- is_correct should be true if score >= 0.7.
- feedback must be warm and pedagogically useful.
- correction: include ONLY if is_correct is false; show the correct answer with English explanation.

Return ONLY the JSON object below — no markdown fences, no extra text.

Required JSON schema:
{_VOICE_EVAL_SCHEMA}
"""


async def evaluate_voice_answer(
    question_data: Dict,
    audio_bytes: bytes,
    mime_type: str = "audio/ogg",
) -> Dict:
    """Transcribe a voice message and evaluate it against the pending question."""
    client = _client_or_raise()
    prompt_text = _build_voice_eval_prompt(question_data)

    contents = [
        types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        types.Part.from_text(text=prompt_text),
    ]

    last_exc: Optional[Exception] = None
    for model in MODELS:
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda m=model: client.models.generate_content(
                    model=m,
                    contents=contents,
                    config=_gen_config(),
                ),
            )
            if model != MODELS[0]:
                logger.info("Voice eval fell back to model: %s", model)
            data = _extract_json(response.text)
            data.setdefault("transcription", "")
            data.setdefault("is_correct", False)
            data.setdefault("score", 0.0)
            data.setdefault("feedback", "")
            return data
        except Exception as exc:
            if _is_quota_error(exc):
                logger.warning("Quota exhausted on %s (voice eval), trying next model.", model)
                last_exc = exc
                continue
            raise
    raise GeminiQuotaExceeded("All models exhausted their daily quota.") from last_exc
