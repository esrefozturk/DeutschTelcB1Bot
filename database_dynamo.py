"""
DynamoDB-backed database layer for AWS Lambda deployment.
Drop-in replacement for database.py — identical public interface.

Tables (names set via env vars):
  USERS_TABLE       — PK: user_id (S)
  SESSIONS_TABLE    — PK: user_id (S)
  PERFORMANCE_TABLE — PK: user_id (S), SK: topic_subtopic (S)
"""

import json
import os
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import boto3
from boto3.dynamodb.conditions import Key

USERS_TABLE       = os.environ.get("USERS_TABLE",       "DeutschBotUsers")
SESSIONS_TABLE    = os.environ.get("SESSIONS_TABLE",    "DeutschBotSessions")
PERFORMANCE_TABLE = os.environ.get("PERFORMANCE_TABLE", "DeutschBotPerformance")
REPORTS_TABLE     = os.environ.get("REPORTS_TABLE",     "DeutschBotReports")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _int(v) -> int:
    return int(v) if v is not None else 0


def _float(v) -> float:
    return float(v) if v is not None else 0.0


class DynamoDatabase:
    def __init__(self):
        ddb = boto3.resource("dynamodb")
        self._users       = ddb.Table(USERS_TABLE)
        self._sessions    = ddb.Table(SESSIONS_TABLE)
        self._performance = ddb.Table(PERFORMANCE_TABLE)
        self._reports     = ddb.Table(REPORTS_TABLE)

    # ── Users ──────────────────────────────────────────────────────────────

    def upsert_user(self, user_id: int, username: str, first_name: str):
        self._users.update_item(
            Key={"user_id": str(user_id)},
            UpdateExpression=(
                "SET username    = :u, "
                "    first_name  = :f, "
                "    created_at  = if_not_exists(created_at, :now), "
                "    last_active = :now"
            ),
            ExpressionAttributeValues={
                ":u":   username,
                ":f":   first_name,
                ":now": _now(),
            },
        )

    def get_all_users(self) -> List[Dict]:
        """Scan all users. Only used for scheduled reminders (low frequency)."""
        result = []
        scan_kwargs: dict = {}
        while True:
            resp = self._users.scan(**scan_kwargs)
            for item in resp.get("Items", []):
                result.append({
                    "user_id":            int(item["user_id"]),
                    "first_name":         item.get("first_name", ""),
                    "last_active":        item.get("last_active", ""),
                    "last_reminder_sent": item.get("last_reminder_sent", ""),
                    "current_streak":     _int(item.get("current_streak", 0)),
                    "is_paused":          _int(item.get("is_paused", 0)),
                })
            if "LastEvaluatedKey" not in resp:
                break
            scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        return result

    def mark_reminder_sent(self, user_id: int):
        self._users.update_item(
            Key={"user_id": str(user_id)},
            UpdateExpression="SET last_reminder_sent = :now",
            ExpressionAttributeValues={":now": _now()},
        )

    def set_paused(self, user_id: int, paused: bool):
        self._users.update_item(
            Key={"user_id": str(user_id)},
            UpdateExpression="SET is_paused = :v",
            ExpressionAttributeValues={":v": int(paused)},
        )

    def update_streak(self, user_id: int) -> tuple[int, bool]:
        """
        Update daily streak. Returns (new_streak, is_new_day).
        is_new_day is True when this is the first answer of a calendar day.
        """
        _today    = datetime.now(timezone.utc).date()
        today     = _today.isoformat()
        yesterday = (_today - timedelta(days=1)).isoformat()

        resp = self._users.get_item(
            Key={"user_id": str(user_id)},
            ProjectionExpression="current_streak, last_streak_date",
        )
        item           = resp.get("Item", {})
        last_date      = item.get("last_streak_date", "")
        current_streak = _int(item.get("current_streak", 0))

        # Always clear is_paused — answering any question resumes reminders.
        self._users.update_item(
            Key={"user_id": str(user_id)},
            UpdateExpression="SET is_paused = :zero",
            ExpressionAttributeValues={":zero": 0},
        )

        if last_date == today:
            return current_streak, False  # already practiced today

        new_streak = (current_streak + 1) if last_date == yesterday else 1
        self._users.update_item(
            Key={"user_id": str(user_id)},
            UpdateExpression="SET current_streak = :s, last_streak_date = :d",
            ExpressionAttributeValues={":s": new_streak, ":d": today},
        )
        return new_streak, True

    def get_user(self, user_id: int) -> Optional[Dict]:
        resp = self._users.get_item(Key={"user_id": str(user_id)})
        item = resp.get("Item")
        if not item:
            return None
        return {
            "user_id":        int(item["user_id"]),
            "username":       item.get("username", ""),
            "first_name":     item.get("first_name", ""),
            "created_at":     item.get("created_at", ""),
            "current_streak": _int(item.get("current_streak", 0)),
            "tier":           item.get("tier", "default"),
        }

    def get_daily_usage(self, user_id: int) -> int:
        """Returns the number of questions asked today (UTC)."""
        today = datetime.now(timezone.utc).date().isoformat()
        resp  = self._users.get_item(
            Key={"user_id": str(user_id)},
            ProjectionExpression="questions_today, last_question_date",
        )
        item = resp.get("Item", {})
        if item.get("last_question_date", "") != today:
            return 0
        return _int(item.get("questions_today", 0))

    def increment_daily_usage(self, user_id: int):
        """Increment today's question count, resetting if it's a new day."""
        today = datetime.now(timezone.utc).date().isoformat()
        resp  = self._users.get_item(
            Key={"user_id": str(user_id)},
            ProjectionExpression="last_question_date",
        )
        last_date = resp.get("Item", {}).get("last_question_date", "")
        if last_date != today:
            self._users.update_item(
                Key={"user_id": str(user_id)},
                UpdateExpression="SET questions_today = :one, last_question_date = :d",
                ExpressionAttributeValues={":one": 1, ":d": today},
            )
        else:
            self._users.update_item(
                Key={"user_id": str(user_id)},
                UpdateExpression="ADD questions_today :one",
                ExpressionAttributeValues={":one": Decimal("1")},
            )

    # ── Sessions / Pending question ─────────────────────────────────────

    def get_pending_question(self, user_id: int) -> Optional[Dict]:
        resp = self._sessions.get_item(Key={"user_id": str(user_id)})
        item = resp.get("Item")
        if not item or not item.get("pending_question"):
            return None
        raw = item["pending_question"]
        return json.loads(raw) if isinstance(raw, str) else raw

    def set_pending_question(self, user_id: int, question: Optional[Dict], count: bool = False):
        payload = json.dumps(question) if question is not None else None
        self._sessions.update_item(
            Key={"user_id": str(user_id)},
            UpdateExpression=(
                "SET pending_question = :q, "
                "    updated_at       = :now, "
                "    questions_sent   = if_not_exists(questions_sent, :zero) + :inc"
            ),
            ExpressionAttributeValues={
                ":q":    payload,
                ":now":  _now(),
                ":zero": 0,
                ":inc":  1 if (question is not None and count) else 0,
            },
        )

    def get_exam_state(self, user_id: int) -> Optional[Dict]:
        resp = self._sessions.get_item(
            Key={"user_id": str(user_id)},
            ProjectionExpression="exam_state",
        )
        item = resp.get("Item")
        if not item or not item.get("exam_state"):
            return None
        raw = item["exam_state"]
        return json.loads(raw) if isinstance(raw, str) else raw

    def set_exam_state(self, user_id: int, state: Optional[Dict]):
        payload = json.dumps(state) if state is not None else None
        self._sessions.update_item(
            Key={"user_id": str(user_id)},
            UpdateExpression="SET exam_state = :s",
            ExpressionAttributeValues={":s": payload},
        )

    def get_questions_sent(self, user_id: int) -> int:
        resp = self._sessions.get_item(
            Key={"user_id": str(user_id)},
            ProjectionExpression="questions_sent",
        )
        item = resp.get("Item")
        return _int(item.get("questions_sent", 0)) if item else 0

    def get_question_cache(self, user_id: int) -> list:
        resp = self._sessions.get_item(
            Key={"user_id": str(user_id)},
            ProjectionExpression="question_cache",
        )
        item = resp.get("Item")
        if not item or not item.get("question_cache"):
            return []
        raw = item["question_cache"]
        try:
            return json.loads(raw) if isinstance(raw, str) else list(raw)
        except Exception:
            return []

    def set_question_cache(self, user_id: int, cache: list):
        payload = json.dumps(cache)
        self._sessions.update_item(
            Key={"user_id": str(user_id)},
            UpdateExpression="SET question_cache = :c",
            ExpressionAttributeValues={":c": payload},
        )

    def get_recent_questions(self, user_id: int) -> list:
        resp = self._sessions.get_item(
            Key={"user_id": str(user_id)},
            ProjectionExpression="recent_questions",
        )
        item = resp.get("Item")
        if not item or not item.get("recent_questions"):
            return []
        raw = item["recent_questions"]
        try:
            return json.loads(raw) if isinstance(raw, str) else list(raw)
        except Exception:
            return []

    def add_recent_question(self, user_id: int, question_text: str, max_size: int = 25):
        snippet = question_text[:100].strip()
        recent  = self.get_recent_questions(user_id)
        if snippet not in recent:
            recent.append(snippet)
        if len(recent) > max_size:
            recent = recent[-max_size:]
        payload = json.dumps(recent)
        self._sessions.update_item(
            Key={"user_id": str(user_id)},
            UpdateExpression="SET recent_questions = :rq",
            ExpressionAttributeValues={":rq": payload},
        )

    # ── Performance ─────────────────────────────────────────────────────

    def get_all_performance(self, user_id: int) -> List[Dict]:
        resp = self._performance.query(
            KeyConditionExpression=Key("user_id").eq(str(user_id))
        )
        result = []
        for item in resp.get("Items", []):
            parts = item["topic_subtopic"].split("#", 1)
            result.append({
                "topic":           parts[0],
                "subtopic":        parts[1] if len(parts) > 1 else "",
                "correct":         _int(item.get("correct",          0)),
                "incorrect":       _int(item.get("incorrect",        0)),
                "total_score":     _float(item.get("total_score",    0.0)),
                "difficulty":      _float(item.get("difficulty",     2.0)),
                "review_interval": _float(item.get("review_interval", 1.0)),
                "last_tested":     item.get("last_tested", ""),
            })
        return result

    def get_topic_performance(self, user_id: int, topic: str, subtopic: str) -> Dict:
        resp = self._performance.get_item(
            Key={
                "user_id":        str(user_id),
                "topic_subtopic": f"{topic}#{subtopic}",
            }
        )
        item = resp.get("Item")
        if not item:
            return {"correct": 0, "incorrect": 0, "difficulty": 2.0, "review_interval": 1.0}
        return {
            "correct":         _int(item.get("correct",          0)),
            "incorrect":       _int(item.get("incorrect",        0)),
            "difficulty":      _float(item.get("difficulty",     2.0)),
            "review_interval": _float(item.get("review_interval", 1.0)),
        }

    def update_performance(
        self,
        user_id: int,
        topic: str,
        subtopic: str,
        is_correct: bool,
        score: float,
        new_difficulty: float,
        review_interval: float,
    ):
        self._performance.update_item(
            Key={
                "user_id":        str(user_id),
                "topic_subtopic": f"{topic}#{subtopic}",
            },
            UpdateExpression=(
                "ADD #c :c_val, #i :i_val, total_score :score "
                "SET difficulty = :diff, review_interval = :ri, last_tested = :now"
            ),
            ExpressionAttributeNames={
                "#c": "correct",
                "#i": "incorrect",
            },
            ExpressionAttributeValues={
                ":c_val": Decimal(str(int(is_correct))),
                ":i_val": Decimal(str(int(not is_correct))),
                ":score": Decimal(str(round(score, 4))),
                ":diff":  Decimal(str(round(new_difficulty, 2))),
                ":ri":    Decimal(str(round(review_interval, 1))),
                ":now":   _now(),
            },
        )

    def get_stats_summary(self, user_id: int) -> Dict[str, Any]:
        rows = self.get_all_performance(user_id)

        total_correct   = sum(r["correct"]     for r in rows)
        total_incorrect = sum(r["incorrect"]   for r in rows)
        total_score     = sum(r["total_score"] for r in rows)
        total           = total_correct + total_incorrect
        accuracy        = round(total_correct / total * 100, 1) if total else 0
        avg_score       = round(total_score    / total * 100, 1) if total else 0

        weak = []
        for r in rows:
            t = r["correct"] + r["incorrect"]
            if t >= 2:
                weak.append({
                    "topic":      r["topic"],
                    "subtopic":   r["subtopic"],
                    "error_rate": r["incorrect"] / t,
                    "avg_score":  r["total_score"] / t,
                })
        weak.sort(key=lambda x: -x["error_rate"])
        weak = weak[:3]

        user      = self.get_user(user_id)
        streak    = user["current_streak"] if user else 0

        return {
            "total_correct":   total_correct,
            "total_incorrect": total_incorrect,
            "total_questions": total,
            "accuracy":        accuracy,
            "avg_score":       avg_score,
            "streak":          streak,
            "weak_areas":      weak,
            "questions_sent":  self.get_questions_sent(user_id),
        }

    # ── Reports ────────────────────────────────────────────────────────────────

    def save_report(self, user_id: int, question: dict, stage: str):
        import uuid
        report_id = str(uuid.uuid4())
        self._reports.put_item(Item={
            "report_id":   report_id,
            "user_id":     str(user_id),
            "reported_at": _now(),
            "stage":       stage,
            "question":    json.dumps(question),
        })
