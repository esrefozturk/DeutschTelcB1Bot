"""Tests for the adaptive learning engine (adaptive.py)."""

import pytest
from datetime import datetime, timedelta, timezone

from adaptive import (
    TOPICS,
    QUESTION_TYPES_BY_TOPIC,
    MIN_DIFFICULTY,
    MAX_DIFFICULTY,
    adjust_difficulty,
    difficulty_label,
    next_review_interval,
    pick_next_params,
    get_subtopic_description,
    all_subtopics_flat,
)


class TestAdjustDifficulty:
    def test_correct_increases_difficulty(self):
        assert adjust_difficulty(2.0, True) == 2.5
        assert adjust_difficulty(2.5, True) == 3.0

    def test_incorrect_decreases_difficulty(self):
        assert adjust_difficulty(3.0, False) == 2.25
        assert adjust_difficulty(2.0, False) == 1.25

    def test_clamped_at_min(self):
        assert adjust_difficulty(1.0, False) == MIN_DIFFICULTY
        assert adjust_difficulty(1.2, False) == MIN_DIFFICULTY

    def test_clamped_at_max(self):
        assert adjust_difficulty(5.0, True) == MAX_DIFFICULTY
        assert adjust_difficulty(4.8, True) == MAX_DIFFICULTY


class TestDifficultyLabel:
    def test_labels(self):
        assert difficulty_label(1.0) == "beginner"
        assert difficulty_label(2.0) == "elementary"
        assert difficulty_label(3.0) == "intermediate"
        assert difficulty_label(4.0) == "upper-intermediate"
        assert difficulty_label(5.0) == "advanced"

    def test_boundaries(self):
        assert difficulty_label(1.5) == "beginner"
        assert difficulty_label(2.5) == "elementary"
        assert difficulty_label(3.5) == "intermediate"
        assert difficulty_label(4.5) == "upper-intermediate"


class TestNextReviewInterval:
    def test_correct_doubles(self):
        assert next_review_interval(1.0, True) == 2.0
        assert next_review_interval(2.0, True) == 4.0

    def test_correct_capped_at_30(self):
        assert next_review_interval(30.0, True) == 30.0
        assert next_review_interval(20.0, True) == 30.0

    def test_incorrect_resets_to_one(self):
        assert next_review_interval(10.0, False) == 1.0
        assert next_review_interval(1.0, False) == 1.0


class TestPickNextParams:
    def test_returns_four_tuple(self):
        topic, subtopic, q_type, difficulty = pick_next_params([])
        assert topic in TOPICS
        assert subtopic in TOPICS[topic]["subtopics"]
        assert q_type in QUESTION_TYPES_BY_TOPIC[topic]
        assert MIN_DIFFICULTY <= difficulty <= MAX_DIFFICULTY

    def test_with_empty_performance_uses_default_difficulty(self):
        for _ in range(20):
            _, _, _, difficulty = pick_next_params([])
            assert difficulty == 2.0

    def test_with_performance_uses_stored_difficulty(self):
        # One subtopic with high difficulty
        perf = [
            {
                "topic": "grammar",
                "subtopic": "cases",
                "correct": 10,
                "incorrect": 2,
                "difficulty": 4.0,
                "review_interval": 2.0,
                "last_tested": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            }
        ]
        seen = set()
        for _ in range(50):
            topic, subtopic, _, difficulty = pick_next_params(perf)
            seen.add((topic, subtopic))
            if (topic, subtopic) == ("grammar", "cases"):
                assert difficulty == 4.0
        # Should sometimes pick other subtopics too
        assert len(seen) >= 2

    def test_all_topics_and_subtopics_valid(self):
        for _ in range(100):
            topic, subtopic, q_type, _ = pick_next_params([])
            assert topic in TOPICS
            assert subtopic in TOPICS[topic]["subtopics"]
            assert q_type in QUESTION_TYPES_BY_TOPIC[topic]


class TestGetSubtopicDescription:
    def test_known_subtopic(self):
        assert "Nominativ" in get_subtopic_description("grammar", "cases")
        assert "Perfekt" in get_subtopic_description("grammar", "verb_conjugation")

    def test_unknown_returns_subtopic_string(self):
        assert get_subtopic_description("grammar", "unknown_thing") == "unknown_thing"


class TestAllSubtopicsFlat:
    def test_returns_tuples(self):
        result = all_subtopics_flat()
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            topic, subtopic = item
            assert topic in TOPICS
            assert subtopic in TOPICS[topic]["subtopics"]

    def test_covers_all_topics(self):
        result = all_subtopics_flat()
        topics_covered = {t for t, _ in result}
        assert topics_covered == set(TOPICS.keys())
