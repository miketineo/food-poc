"""Validated food catalogue matcher.

Decides whether an incoming food record should be AUTO_ACCEPT-ed onto a
canonical food, sent to REVIEW, or treated as NEW.

Assumptions:
- Canonical nutrition is per 100 g; incoming nutrition is per serving.
- Serving units convertible to grams: g/gram(s)/gr, kg, oz. Anything else
  (ml, scoops...) means nutrition is missing evidence, not contradictory.
- Barcodes are compared digits-only and exactly (leading zeros preserved).
- Wrong merges cost more than reviews, so AUTO_ACCEPT requires either a
  barcode match with 100% agreement on every other comparable signal, or
  (without barcode) a near-exact name plus agreeing nutrition.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rapidfuzz import fuzz

import ml

AUTO_ACCEPT = "AUTO_ACCEPT"
REVIEW = "REVIEW"
NEW = "NEW"

MATCH = "match"
PARTIAL = "partial"
CONFLICT = "conflict"
MISSING = "missing"

# Tunables: kept together so a requirement change is a one-line edit.
WEIGHTS = {"barcode": 0.35, "name": 0.30, "nutrition": 0.25, "brand": 0.10}
NAME_MATCH_SIM = 0.95
NAME_CONFLICT_SIM = 0.55
BRAND_MATCH_SIM = 0.90
NUTRITION_REL_TOL = 0.10
NUTRITION_ABS_TOL = {"calories": 10.0, "protein": 0.5, "carbs": 0.5, "fat": 0.5}
NEW_FLOOR = 0.45
AMBIGUITY_MARGIN = 0.10
SEMANTIC_RANK_CAP = 0.90  # semantic similarity aids ranking, never auto-accept
ML_BRAKE = 0.80           # model may veto an accept below this, never force one

MASS_TO_G = {"g": 1.0, "gram": 1.0, "grams": 1.0, "gr": 1.0, "kg": 1000.0, "oz": 28.3495}
NUTRIENTS = ("calories", "protein", "carbs", "fat")
UNIT_NOISE = set(MASS_TO_G) | {"ml", "l", "kcal", "cal"}


@dataclass
class MatchResult:
    canonical_food_id: str | None
    decision: str
    confidence: float | None
    reasons: list[str] = field(default_factory=list)


# --- normalization -----------------------------------------------------------

def _tokens(text: str) -> list[str]:
    text = re.sub(r"[^a-z0-9]+", " ", str(text).lower())
    text = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", text)
    return text.split()


def normalize_name(name: str | None) -> str:
    toks = _tokens(name or "")
    out: list[str] = []
    i = 0
    while i < len(toks):
        if toks[i].isdigit() and i + 1 < len(toks) and toks[i + 1] in UNIT_NOISE:
            i += 2  # drop "40 grams" style weight noise, keep bare numbers
            continue
        if toks[i] in UNIT_NOISE:
            i += 1
            continue
        out.append(toks[i])
        i += 1
    return " ".join(out)


def normalize_barcode(barcode) -> str | None:
    digits = re.sub(r"\D", "", str(barcode or ""))
    return digits or None


def _normalize_unit(unit) -> str | None:
    if not unit:
        return None
    return str(unit).strip().lower().rstrip(".")


def to_per_100g(record: dict) -> dict | None:
    """Convert per-serving nutrition to per-100g; None when not a mass serving."""
    qty = record.get("serving_quantity")
    unit = _normalize_unit(record.get("serving_unit"))
    if not qty or unit not in MASS_TO_G:
        return None
    grams = float(qty) * MASS_TO_G[unit]
    if grams <= 0:
        return None
    factor = 100.0 / grams
    out = {k: record[k] * factor for k in NUTRIENTS if record.get(k) is not None}
    return out or None


def name_similarity(a: str | None, b: str | None) -> float | None:
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return None
    return max(fuzz.ratio(na, nb), fuzz.token_set_ratio(na, nb)) / 100.0


# --- per-candidate evidence --------------------------------------------------

@dataclass
class _Candidate:
    food: dict
    score: float = 0.0
    states: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.food["id"]

    @property
    def conflicts(self) -> list[str]:
        return [s for s, st in self.states.items() if st == CONFLICT]


def _compare_barcode(incoming: dict, food: dict) -> tuple[str, float, str]:
    a, b = normalize_barcode(incoming.get("barcode")), normalize_barcode(food.get("barcode"))
    if not a or not b:
        return MISSING, 0.0, "barcode absent on one or both records: skipped (missing evidence, not contradictory)"
    if a == b:
        return MATCH, 1.0, f"barcode {a} matches {food['id']}"
    return CONFLICT, 0.0, f"barcode conflict: incoming {a} vs {food['id']} {b}"


def _compare_name(incoming: dict, food: dict) -> tuple[str, float, str]:
    sim = name_similarity(incoming.get("name"), food.get("name"))
    if sim is None:
        return MISSING, 0.0, "name absent on one or both records"
    note = f"name similarity {sim:.2f} ('{incoming.get('name')}' vs '{food.get('name')}')"
    value = sim
    sem = ml.semantic_similarity(incoming.get("name"), food.get("name"))
    if sem is not None and min(sem, SEMANTIC_RANK_CAP) > value:
        value = min(sem, SEMANTIC_RANK_CAP)
        note += f"; semantic similarity {sem:.2f} used for ranking (capped, cannot auto-accept)"
    # states come from lexical similarity only: embeddings rank, rules decide
    if sim >= NAME_MATCH_SIM:
        return MATCH, value, note
    if sim <= NAME_CONFLICT_SIM:
        return CONFLICT, value, note + ": names conflict"
    return PARTIAL, value, note + ": similar but not near-exact"


def _compare_brand(incoming: dict, food: dict) -> tuple[str, float, str]:
    sim = name_similarity(incoming.get("brand"), food.get("brand"))
    if sim is None:
        return MISSING, 0.0, "brand absent on one or both records: skipped (missing evidence, not contradictory)"
    if sim >= BRAND_MATCH_SIM:
        return MATCH, 1.0, f"brand agrees ('{food.get('brand')}')"
    return CONFLICT, 0.0, f"brand conflict: '{incoming.get('brand')}' vs '{food.get('brand')}'"


def _compare_nutrition(incoming: dict, food: dict) -> tuple[str, float, str]:
    per100 = to_per_100g(incoming)
    pairs: dict[str, tuple[float, float]] = {}
    if per100:
        for k in NUTRIENTS:
            canon_val = food.get(f"{k}_per_100g")
            if k in per100 and canon_val is not None:
                pairs[k] = (per100[k], float(canon_val))
    if not pairs:
        return MISSING, 0.0, "nutrition not comparable (no mass serving or values): missing evidence, not contradictory"
    off = [f"{k} per 100 g differ: incoming {a:.1f} vs canonical {b:g}"
           for k, (a, b) in pairs.items()
           if abs(a - b) > max(NUTRITION_ABS_TOL[k], NUTRITION_REL_TOL * max(abs(a), abs(b)))]
    if off:
        return CONFLICT, 0.0, "nutrition conflict: " + "; ".join(off)
    qty, unit = incoming.get("serving_quantity"), incoming.get("serving_unit")
    return MATCH, 1.0, f"nutrition per 100 g within tolerance (converted from {qty} {unit} serving)"


def _score_candidate(incoming: dict, food: dict) -> _Candidate:
    cand = _Candidate(food=food)
    signals = {
        "barcode": _compare_barcode,
        "name": _compare_name,
        "brand": _compare_brand,
        "nutrition": _compare_nutrition,
    }
    num = den = 0.0
    for signal, compare in signals.items():
        state, value, note = compare(incoming, food)
        cand.states[signal] = state
        cand.notes.append(note)
        if state != MISSING:
            num += WEIGHTS[signal] * value
            den += WEIGHTS[signal]
    cand.score = num / den if den else 0.0
    return cand


# --- decision ----------------------------------------------------------------

def _auto_accept_criteria(cand: _Candidate) -> bool:
    s = cand.states
    if cand.conflicts:
        return False
    if s["barcode"] == MATCH:
        # barcode alone is never sufficient: every other comparable signal must agree
        return all(st in (MATCH, MISSING) for st in s.values())
    return s["name"] == MATCH and s["nutrition"] == MATCH and s["brand"] in (MATCH, MISSING)


def match_food(incoming: dict, catalogue: list[dict]) -> MatchResult:
    if not catalogue:
        return MatchResult(None, NEW, None, ["catalogue is empty: nothing to match against"])

    candidates = sorted(
        (_score_candidate(incoming, food) for food in catalogue),
        key=lambda c: c.score, reverse=True,
    )
    best = candidates[0]
    barcode_hits = [c for c in candidates if c.states["barcode"] == MATCH]
    reasons = list(best.notes)

    if best.score < NEW_FLOOR and not barcode_hits:
        return MatchResult(None, NEW, None, [
            f"best candidate {best.id} ('{best.food.get('name')}') scored {best.score:.2f}, below the {NEW_FLOOR} floor",
            "no barcode match anywhere in the catalogue",
            "treating record as a new food",
        ])

    prob = ml.match_probability(incoming, best.food, catalogue)
    if prob is not None:
        reasons.append(f"learned match probability {prob:.2f} (logistic model trained on catalogue-derived pairs)")
    confidence = round(prob, 2) if prob is not None else round(best.score, 2)

    foreign = next((h for h in barcode_hits if h.id != best.id), None)
    if foreign:
        reasons.append(
            f"incoming barcode belongs to {foreign.id} ('{foreign.food.get('name')}') "
            f"while other evidence favours {best.id}: contradictory evidence"
        )
        return MatchResult(best.id, REVIEW, confidence, reasons)

    if best.conflicts:
        reasons.append(f"contradictory evidence on {', '.join(best.conflicts)}: routed to review")
        return MatchResult(best.id, REVIEW, confidence, reasons)

    runner_up = candidates[1] if len(candidates) > 1 else None
    if runner_up and runner_up.score >= NEW_FLOOR and best.score - runner_up.score < AMBIGUITY_MARGIN:
        reasons.append(
            f"ambiguous: {runner_up.id} ('{runner_up.food.get('name')}') scores {runner_up.score:.2f}, "
            f"within {AMBIGUITY_MARGIN} of best {best.score:.2f}"
        )
        return MatchResult(best.id, REVIEW, confidence, reasons)

    if _auto_accept_criteria(best):
        if prob is not None and prob < ML_BRAKE:
            reasons.append(f"model probability {prob:.2f} is below the {ML_BRAKE} brake: downgraded to review")
            return MatchResult(best.id, REVIEW, confidence, reasons)
        reasons.append("all available evidence agrees: auto-accepted")
        return MatchResult(best.id, AUTO_ACCEPT, confidence, reasons)

    reasons.append("evidence supports a match but is not strong enough to auto-accept")
    return MatchResult(best.id, REVIEW, confidence, reasons)


def main() -> None:
    here = Path(__file__).parent
    incoming = json.loads((here / "incoming_foods.json").read_text())
    catalogue = json.loads((here / "canonical_foods.json").read_text())
    for record in incoming:
        result = match_food(record, catalogue)
        print(json.dumps({"source_id": record.get("source_id"), **asdict(result)}, indent=2))


if __name__ == "__main__":
    main()
