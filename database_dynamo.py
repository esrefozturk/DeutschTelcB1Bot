"""
DynamoDB-backed database layer for AWS Lambda deployment.
Drop-in replacement for database.py — identical public interface.

Tables (names set via env vars):
  USERS_TABLE       — PK: user_id (S)
  SESSIONS_TABLE    — PK: user_id (S)
  PERFORMANCE_TABLE — PK: user_id (S), SK: topic_subtopic (S)
"""

import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import boto3
from boto3.dynamodb.conditions import Key

USERS_TABLE       = os.environ.get("USERS_TABLE",       "DeutschBotUsers")
SESSIONS_TABLE    = os.environ.get("SESSIONS_TABLE",    "DeutschBotSessions")
PERFORMANCE_TABLE = os.environ.get("PERFORMANCE_TABLE", "DeutschBotPerformance")


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

    # ── Users ──────────────────────────────────────────────────────────────

    def upsert_user(self, user_id: int, username: str, first_name: str):
        self._users.update_item(
            Key={"user_id": str(user_id)},
            UpdateExpression=(
                "SET username   = :u, "
                "    first_name = :f, "
                "    created_at = if_not_exists(created_at, :now), "
                "    is_paused  = if_not_exists(is_paused,  :zero)"
            ),
            ExpressionAttributeValues={
                ":u":    username,
                ":f":    first_name,
                ":now":  _now(),
                ":zero": 0,
            },
        )

    def get_user(self, user_id: int) -> Optional[Dict]:
        resp = self._users.get_item(Key={"user_id": str(user_id)})
        item = resp.get("Item")
        if not item:
            return None
        return {
            "user_id":    int(item["user_id"]),
            "username":   item.get("username", ""),
            "first_name": item.get("first_name", ""),
            "is_paused":  _int(item.get("is_paused", 0)),
            "created_at": item.get("created_at", ""),
        }

    def set_paused(self, user_id: int, paused: bool):
        self._users.update_item(
            Key={"user_id": str(user_id)},
            UpdateExpression="SET is_paused = :v",
            ExpressionAttributeValues={":v": int(paused)},
        )

    # ── Sessions / Pending question ─────────────────────────────────────

    def get_pending_question(self, user_id: int) -> Optional[Dict]:
        import json
        resp = self._sessions.get_item(Key={"user_id": str(user_id)})
        item = resp.get("Item")
        if not item or not item.get("pending_question"):
            return None
        raw = item["pending_question"]
        return json.loads(raw) if isinstance(raw, str) else raw

    def set_pending_question(self, user_id: int, question: Optional[Dict]):
        import json
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
                ":inc":  1 if question is not None else 0,
            },
        )

    def get_questions_sent(self, user_id: int) -> int:
        resp = self._sessions.get_item(
            Key={"user_id": str(user_id)},
            ProjectionExpression="questions_sent",
        )
        item = resp.get("Item")
        return _int(item.get("questions_sent", 0)) if item else 0

    # ── Performance ─────────────────────────────────────────────────────

    def get_all_performance(self, user_id: int) -> List[Dict]:
        resp = self._performance.query(
            KeyConditionExpression=Key("user_id").eq(str(user_id))
        )
        result = []
        for item in resp.get("Items", []):
            parts = item["topic_subtopic"].split("#", 1)
            result.append({
                "topic":    parts[0],
                "subtopic": parts[1] if len(parts) > 1 else "",
                "correct":   _int(item.get("correct",   0)),
                "incorrect": _int(item.get("incorrect", 0)),
                "difficulty": _float(item.get("difficulty", 2.0)),
                "last_tested": item.get("last_tested", ""),
            })
        return result

    def get_topic_performance(self, user_id: int, topic: str, subtopic: str) -> Dict:
        resp = self._performance.get_item(
            Key={
                "user_id":       str(user_id),
                "topic_subtopic": f"{topic}#{subtopic}",
            }
        )
        item = resp.get("Item")
        if not item:
            return {"correct": 0, "incorrect": 0, "difficulty": 2.0}
        return {
            "correct":    _int(item.get("correct",   0)),
            "incorrect":  _int(item.get("incorrect", 0)),
            "difficulty": _float(item.get("difficulty", 2.0)),
        }

    def update_performance(
        self,
        user_id: int,
        topic: str,
        subtopic: str,
        is_correct: bool,
        new_difficulty: float,
    ):
        # ADD atomically increments (creates with value 0+inc if item is new)
        self._performance.update_item(
            Key={
                "user_id":        str(user_id),
                "topic_subtopic": f"{topic}#{subtopic}",
            },
            UpdateExpression=(
                "ADD #c :c_val, #i :i_val "
                "SET difficulty = :diff, last_tested = :now"
            ),
            ExpressionAttributeNames={
                "#c": "correct",
                "#i": "incorrect",
            },
            ExpressionAttributeValues={
                ":c_val": Decimal(str(int(is_correct))),
                ":i_val": Decimal(str(int(not is_correct))),
                ":diff":  Decimal(str(round(new_difficulty, 2))),
                ":now":   _now(),
            },
        )

    def get_stats_summary(self, user_id: int) -> Dict[str, Any]:
        rows = self.get_all_performance(user_id)

        total_correct   = sum(r["correct"]   for r in rows)
        total_incorrect = sum(r["incorrect"] for r in rows)
        total           = total_correct + total_incorrect
        accuracy        = round(total_correct / total * 100, 1) if total else 0

        # Weak areas: subtopics with ≥2 attempts, sorted by error rate
        weak = []
        for r in rows:
            t = r["correct"] + r["incorrect"]
            if t >= 2:
                weak.append({
                    "topic":      r["topic"],
                    "subtopic":   r["subtopic"],
                    "error_rate": r["incorrect"] / t,
                })
        weak.sort(key=lambda x: -x["error_rate"])
        weak = weak[:3]

        return {
            "total_correct":   total_correct,
            "total_incorrect": total_incorrect,
            "total_questions": total,
            "accuracy":        accuracy,
            "weak_areas":      weak,
            "questions_sent":  self.get_questions_sent(user_id),
        }
