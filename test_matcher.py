import json
from pathlib import Path

import pytest

from matcher import AUTO_ACCEPT, NEW, REVIEW, match_food, to_per_100g

HERE = Path(__file__).parent
CATALOGUE = json.loads((HERE / "canonical_foods.json").read_text())
INCOMING = {r["source_id"]: r for r in json.loads((HERE / "incoming_foods.json").read_text())}


def reasons_mention(result, word):
    return any(word.lower() in r.lower() for r in result.reasons)


# --- normalization -----------------------------------------------------------

def test_per_serving_converts_to_per_100g():
    per100 = to_per_100g(INCOMING["m-100"])  # 100 kcal / 40 "grams"
    assert per100["calories"] == pytest.approx(250)
    assert per100["carbs"] == pytest.approx(62.5)


def test_non_mass_serving_unit_cannot_convert():
    record = dict(INCOMING["m-100"], serving_unit="ml")
    assert to_per_100g(record) is None


# --- decisions on the supplied fixtures --------------------------------------

def test_barcode_and_full_agreement_auto_accepts():
    result = match_food(INCOMING["m-100"], CATALOGUE)
    assert result.decision == AUTO_ACCEPT
    assert result.canonical_food_id == "food-001"
    assert result.confidence >= 0.9
    assert reasons_mention(result, "barcode")


def test_barcode_conflict_with_name_evidence_goes_to_review():
    # w-300: name/brand say Maurten Gel 100 but barcode belongs to food-003
    result = match_food(INCOMING["w-300"], CATALOGUE)
    assert result.decision == REVIEW
    assert result.canonical_food_id is not None
    assert reasons_mention(result, "barcode")
    assert reasons_mention(result, "food-003")


def test_moderate_name_match_without_barcode_goes_to_review():
    # u-200: banana medium vs "Banana, raw" - nutrition agrees but name is not
    # near-exact; asymmetric merge cost means a human confirms it
    result = match_food(INCOMING["u-200"], CATALOGUE)
    assert result.decision == REVIEW
    assert result.canonical_food_id == "food-002"
    assert result.confidence is not None


def test_unknown_food_is_new():
    result = match_food(INCOMING["u-400"], CATALOGUE)
    assert result.decision == NEW
    assert result.canonical_food_id is None
    assert result.confidence is None


# --- missing vs contradictory evidence ---------------------------------------

def test_missing_barcode_is_not_penalized():
    incoming = {
        "source": "test", "source_id": "t-1",
        "name": "Maurten Gel 100", "brand": "Maurten", "barcode": None,
        "serving_quantity": 40, "serving_unit": "g",
        "calories": 100, "protein": 0, "carbs": 25, "fat": 0,
    }
    result = match_food(incoming, CATALOGUE)
    assert result.decision == AUTO_ACCEPT
    assert result.canonical_food_id == "food-001"
    assert not reasons_mention(result, "conflict")


def test_nutrition_contradiction_blocks_barcode_auto_accept():
    incoming = dict(INCOMING["m-100"], calories=500)  # 1250 kcal/100g vs 250
    result = match_food(incoming, CATALOGUE)
    assert result.decision == REVIEW
    assert reasons_mention(result, "calories")


def test_matching_barcode_never_yields_new():
    incoming = {
        "source": "test", "source_id": "t-2",
        "name": "Completely Different Product", "brand": "Acme", "barcode": "73187377",
        "serving_quantity": 30, "serving_unit": "g",
        "calories": 900, "protein": 50, "carbs": 1, "fat": 80,
    }
    result = match_food(incoming, CATALOGUE)
    assert result.decision == REVIEW


# --- ambiguity ---------------------------------------------------------------

def test_ambiguous_candidates_go_to_review():
    catalogue = [
        {"id": "gel-lemon", "name": "Energy Gel Lemon", "brand": "Example Sports",
         "barcode": None, "serving_quantity": 45, "serving_unit": "g",
         "calories_per_100g": 240, "protein_per_100g": 0,
         "carbs_per_100g": 60, "fat_per_100g": 0},
        {"id": "gel-lime", "name": "Energy Gel Lime", "brand": "Example Sports",
         "barcode": None, "serving_quantity": 45, "serving_unit": "g",
         "calories_per_100g": 240, "protein_per_100g": 0,
         "carbs_per_100g": 60, "fat_per_100g": 0},
    ]
    incoming = {
        "source": "test", "source_id": "t-3",
        "name": "Energy Gel Lemon", "brand": "Example Sports", "barcode": None,
        "serving_quantity": 45, "serving_unit": "g",
        "calories": 108, "protein": 0, "carbs": 27, "fat": 0,
    }
    result = match_food(incoming, catalogue)
    assert result.decision == REVIEW
    assert reasons_mention(result, "ambiguous") or reasons_mention(result, "close")


# --- output contract ---------------------------------------------------------

def test_every_decision_has_reasons():
    for record in INCOMING.values():
        result = match_food(record, CATALOGUE)
        assert result.reasons, f"no reasons for {record['source_id']}"
        assert result.decision in {AUTO_ACCEPT, REVIEW, NEW}


def test_empty_catalogue_is_new():
    result = match_food(INCOMING["m-100"], [])
    assert result.decision == NEW
    assert result.canonical_food_id is None
