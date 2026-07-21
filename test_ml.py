import json
from pathlib import Path

import ml
from matcher import AUTO_ACCEPT, REVIEW, match_food

HERE = Path(__file__).parent
CATALOGUE = json.loads((HERE / "canonical_foods.json").read_text())
INCOMING = {r["source_id"]: r for r in json.loads((HERE / "incoming_foods.json").read_text())}


# --- learned probability model -----------------------------------------------

def test_perfect_pair_outscores_mismatched_pair():
    good = ml.match_probability(INCOMING["m-100"], CATALOGUE[0], CATALOGUE)
    bad = ml.match_probability(INCOMING["u-400"], CATALOGUE[0], CATALOGUE)
    assert good is not None and bad is not None
    assert good > 0.9
    assert bad < 0.5
    assert good > bad


def test_training_is_deterministic():
    ml._MODEL_CACHE.clear()
    first = ml.match_probability(INCOMING["m-100"], CATALOGUE[0], CATALOGUE)
    ml._MODEL_CACHE.clear()
    second = ml.match_probability(INCOMING["m-100"], CATALOGUE[0], CATALOGUE)
    assert first == second


def test_single_food_catalogue_has_no_model():
    assert ml.match_probability(INCOMING["m-100"], CATALOGUE[0], CATALOGUE[:1]) is None


def test_confidence_is_model_probability():
    result = match_food(INCOMING["m-100"], CATALOGUE)
    prob = ml.match_probability(INCOMING["m-100"], CATALOGUE[0], CATALOGUE)
    assert result.confidence == round(prob, 2)


# --- safety invariants: ML can veto, never force a merge ---------------------

def test_semantic_similarity_alone_cannot_auto_accept(monkeypatch):
    monkeypatch.setattr(ml, "semantic_similarity", lambda a, b: 1.0)
    incoming = {
        "source": "test", "source_id": "t-sem",
        "name": "Totally Different Words", "brand": "Maurten", "barcode": None,
        "serving_quantity": 40, "serving_unit": "g",
        "calories": 100, "protein": 0, "carbs": 25, "fat": 0,
    }
    result = match_food(incoming, CATALOGUE)
    assert result.decision != AUTO_ACCEPT


def test_low_model_probability_downgrades_accept_to_review(monkeypatch):
    monkeypatch.setattr(ml, "match_probability", lambda inc, food, cat: 0.2)
    result = match_food(INCOMING["m-100"], CATALOGUE)  # otherwise a clean accept
    assert result.decision == REVIEW
    assert any("probability" in r.lower() for r in result.reasons)


def test_high_model_probability_cannot_rescue_a_conflict(monkeypatch):
    monkeypatch.setattr(ml, "match_probability", lambda inc, food, cat: 0.99)
    result = match_food(INCOMING["w-300"], CATALOGUE)  # barcode/name contradiction
    assert result.decision == REVIEW
