"""Tests for the SQLite database layer (database.py)."""

import tempfile
import os

import pytest

from database import Database


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        yield Database(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


class TestUsers:
    def test_upsert_user_and_get_user(self, db):
        db.upsert_user(12345, "alice", "Alice")
        user = db.get_user(12345)
        assert user is not None
        assert user["user_id"] == 12345
        assert user["username"] == "alice"
        assert user["first_name"] == "Alice"

    def test_upsert_updates_existing(self, db):
        db.upsert_user(1, "old", "Old")
        db.upsert_user(1, "new_username", "New Name")
        user = db.get_user(1)
        assert user["username"] == "new_username"
        assert user["first_name"] == "New Name"

    def test_get_user_missing_returns_none(self, db):
        assert db.get_user(99999) is None

    def test_update_streak_first_day(self, db):
        db.upsert_user(1, "u", "U")
        streak, is_new = db.update_streak(1)
        assert streak == 1
        assert is_new is True

    def test_update_streak_same_day_no_increase(self, db):
        db.upsert_user(1, "u", "U")
        db.update_streak(1)
        streak, is_new = db.update_streak(1)
        assert streak == 1
        assert is_new is False


class TestSessions:
    def test_set_and_get_pending_question(self, db):
        db.upsert_user(1, "u", "U")
        q = {"question": "Test?", "correct_answer": "A", "topic": "grammar", "subtopic": "cases"}
        db.set_pending_question(1, q)
        got = db.get_pending_question(1)
        assert got is not None
        assert got["question"] == "Test?"
        assert got["correct_answer"] == "A"

    def test_get_pending_question_empty_returns_none(self, db):
        db.upsert_user(1, "u", "U")
        assert db.get_pending_question(1) is None

    def test_set_pending_question_none_clears(self, db):
        db.upsert_user(1, "u", "U")
        db.set_pending_question(1, {"question": "x"})
        db.set_pending_question(1, None)
        assert db.get_pending_question(1) is None

    def test_question_cache(self, db):
        db.upsert_user(1, "u", "U")
        assert db.get_question_cache(1) == []
        cache = [{"question": "Q1"}, {"question": "Q2"}]
        db.set_question_cache(1, cache)
        assert db.get_question_cache(1) == cache

    def test_recent_questions(self, db):
        db.upsert_user(1, "u", "U")
        assert db.get_recent_questions(1) == []
        db.add_recent_question(1, "First question text")
        db.add_recent_question(1, "Second question text")
        recent = db.get_recent_questions(1)
        assert "First question" in recent[0]
        assert "Second question" in recent[1]


class TestPerformance:
    def test_record_and_get_performance(self, db):
        db.upsert_user(1, "u", "U")
        perf = db.get_all_performance(1)
        assert perf == []

        db.update_performance(
            user_id=1,
            topic="grammar",
            subtopic="cases",
            is_correct=True,
            score=1.0,
            new_difficulty=2.5,
            review_interval=2.0,
        )
        perf = db.get_all_performance(1)
        assert len(perf) == 1
        assert perf[0]["topic"] == "grammar"
        assert perf[0]["subtopic"] == "cases"
        assert perf[0]["correct"] == 1
        assert perf[0]["incorrect"] == 0
        assert perf[0]["difficulty"] == 2.5

    def test_update_performance_increments(self, db):
        db.upsert_user(1, "u", "U")
        db.update_performance(1, "grammar", "cases", True, 1.0, 2.5, 2.0)
        db.update_performance(1, "grammar", "cases", False, 0.0, 1.75, 1.0)
        perf = db.get_topic_performance(1, "grammar", "cases")
        assert perf["correct"] == 1
        assert perf["incorrect"] == 1


class TestStatsSummary:
    def test_empty_user_stats(self, db):
        db.upsert_user(1, "u", "U")
        s = db.get_stats_summary(1)
        assert s["total_questions"] == 0
        assert s["accuracy"] == 0
        assert s["avg_score"] == 0
        assert s["streak"] == 0
        assert s["weak_areas"] == []

    def test_stats_after_answers(self, db):
        db.upsert_user(1, "u", "U")
        db.update_performance(1, "grammar", "cases", True, 1.0, 2.5, 2.0)
        db.update_performance(1, "grammar", "cases", True, 0.8, 3.0, 4.0)
        db.update_performance(1, "grammar", "cases", False, 0.0, 2.25, 1.0)
        s = db.get_stats_summary(1)
        assert s["total_questions"] == 3
        assert s["total_correct"] == 2
        assert s["total_incorrect"] == 1
        assert s["accuracy"] == round(2 / 3 * 100, 1)
        assert "weak_areas" in s
        assert "questions_sent" in s
