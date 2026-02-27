import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, Any


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id     INTEGER PRIMARY KEY,
                    username    TEXT,
                    first_name  TEXT,
                    created_at  TEXT DEFAULT (datetime('now')),
                    is_paused   INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    user_id          INTEGER PRIMARY KEY,
                    pending_question TEXT,           -- JSON blob of the current question
                    questions_sent   INTEGER DEFAULT 0,
                    updated_at       TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS performance (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     INTEGER NOT NULL,
                    topic       TEXT NOT NULL,
                    subtopic    TEXT NOT NULL,
                    correct     INTEGER DEFAULT 0,
                    incorrect   INTEGER DEFAULT 0,
                    total_score REAL DEFAULT 0.0,
                    difficulty  REAL DEFAULT 2.0,   -- 1.0 – 5.0
                    last_tested TEXT,
                    UNIQUE(user_id, topic, subtopic)
                );
            """)
            # Migration: add total_score to existing databases
            try:
                conn.execute("ALTER TABLE performance ADD COLUMN total_score REAL DEFAULT 0.0")
            except Exception:
                pass

    # ── Users ──────────────────────────────────────────────────────────────

    def upsert_user(self, user_id: int, username: str, first_name: str):
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO users (user_id, username, first_name)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET username=excluded.username,
                                                   first_name=excluded.first_name
                """,
                (user_id, username, first_name),
            )

    def get_user(self, user_id: int) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            return dict(row) if row else None

    def set_paused(self, user_id: int, paused: bool):
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET is_paused = ? WHERE user_id = ?",
                (int(paused), user_id),
            )

    # ── Sessions / Pending question ─────────────────────────────────────

    def get_pending_question(self, user_id: int) -> Optional[Dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT pending_question FROM sessions WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row and row["pending_question"]:
                return json.loads(row["pending_question"])
            return None

    def set_pending_question(self, user_id: int, question: Optional[Dict]):
        payload = json.dumps(question) if question else None
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO sessions (user_id, pending_question, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(user_id) DO UPDATE SET pending_question=excluded.pending_question,
                                                   questions_sent=questions_sent + 1,
                                                   updated_at=excluded.updated_at
                """,
                (user_id, payload),
            )

    def get_questions_sent(self, user_id: int) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT questions_sent FROM sessions WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            return row["questions_sent"] if row else 0

    # ── Performance ─────────────────────────────────────────────────────

    def get_all_performance(self, user_id: int) -> list[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM performance WHERE user_id = ?", (user_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_topic_performance(self, user_id: int, topic: str, subtopic: str) -> Dict:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM performance
                WHERE user_id = ? AND topic = ? AND subtopic = ?
                """,
                (user_id, topic, subtopic),
            ).fetchone()
            if row:
                return dict(row)
            return {"correct": 0, "incorrect": 0, "difficulty": 2.0}

    def update_performance(
        self,
        user_id: int,
        topic: str,
        subtopic: str,
        is_correct: bool,
        score: float,
        new_difficulty: float,
    ):
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO performance (user_id, topic, subtopic, correct, incorrect, total_score, difficulty, last_tested)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(user_id, topic, subtopic) DO UPDATE SET
                    correct     = correct     + ?,
                    incorrect   = incorrect   + ?,
                    total_score = total_score + ?,
                    difficulty  = ?,
                    last_tested = datetime('now')
                """,
                (
                    user_id, topic, subtopic,
                    int(is_correct), int(not is_correct), score, new_difficulty,
                    int(is_correct), int(not is_correct), score, new_difficulty,
                ),
            )

    def get_stats_summary(self, user_id: int) -> Dict[str, Any]:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT
                    SUM(correct)     AS total_correct,
                    SUM(incorrect)   AS total_incorrect,
                    SUM(total_score) AS total_score
                FROM performance WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
            total_correct   = row["total_correct"]   or 0
            total_incorrect = row["total_incorrect"] or 0
            total_score     = row["total_score"]     or 0.0
            total           = total_correct + total_incorrect
            accuracy        = round(total_correct / total * 100, 1) if total else 0
            avg_score       = round(total_score / total * 100, 1) if total else 0

            # Weakest topics
            weak = conn.execute(
                """
                SELECT topic, subtopic,
                       CAST(incorrect AS REAL) / (correct + incorrect) AS error_rate,
                       total_score / (correct + incorrect) AS avg_score
                FROM performance
                WHERE user_id = ? AND (correct + incorrect) >= 2
                ORDER BY error_rate DESC
                LIMIT 3
                """,
                (user_id,),
            ).fetchall()

            return {
                "total_correct":   total_correct,
                "total_incorrect": total_incorrect,
                "total_questions": total,
                "accuracy":        accuracy,
                "avg_score":       avg_score,
                "weak_areas":      [dict(r) for r in weak],
                "questions_sent":  self.get_questions_sent(user_id),
            }
