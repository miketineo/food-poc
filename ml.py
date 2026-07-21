"""ML layer for the food matcher.

Two capabilities, both subordinate to matcher's deterministic guardrails:
- semantic_similarity: embedding cosine over names (model2vec, optional).
  Used only to improve candidate ranking; capped so it can never satisfy
  the lexical auto-accept threshold.
- match_probability: calibrated P(same food) from a numpy logistic
  regression trained per catalogue on synthetic pairs (perturbed positives,
  cross-pair and barcode-swap hard negatives). It supplies confidence and
  can veto an auto-accept, never force one.
"""
from __future__ import annotations

import numpy as np

TRAINING_SEED = 42
POSITIVES_PER_FOOD = 40
NEGATIVES_PER_PAIR = 15
HARD_NEGATIVES_PER_PAIR = 5
LEARNING_RATE = 0.5
EPOCHS = 2000
L2 = 1e-3

_MODEL_CACHE: dict[tuple, "_LogisticModel | None"] = {}
_EMBEDDER = None
_EMBEDDER_LOADED = False


# --- optional semantic backend ------------------------------------------------

def _load_embedder():
    global _EMBEDDER, _EMBEDDER_LOADED
    if not _EMBEDDER_LOADED:
        _EMBEDDER_LOADED = True
        try:
            from model2vec import StaticModel
            _EMBEDDER = StaticModel.from_pretrained("minishlab/potion-base-8M")
        except Exception:
            _EMBEDDER = None
    return _EMBEDDER


def semantic_similarity(a: str | None, b: str | None) -> float | None:
    """Cosine similarity of name embeddings; None when backend unavailable."""
    if not a or not b:
        return None
    model = _load_embedder()
    if model is None:
        return None
    va, vb = model.encode([str(a), str(b)])
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return None
    return float(np.dot(va, vb) / denom)


# --- feature extraction -------------------------------------------------------

def _features(incoming: dict, food: dict) -> np.ndarray:
    import matcher

    name_state, name_value, _ = matcher._compare_name(incoming, food)
    states = {
        "barcode": matcher._compare_barcode(incoming, food)[0],
        "brand": matcher._compare_brand(incoming, food)[0],
        "nutrition": matcher._compare_nutrition(incoming, food)[0],
    }

    def one_hot(state):
        return [float(state == matcher.MATCH), float(state == matcher.CONFLICT),
                float(state == matcher.MISSING)]

    missing = sum(1 for s in [name_state, *states.values()] if s == matcher.MISSING)
    return np.array([
        name_value,
        float(name_state == matcher.CONFLICT),
        *one_hot(states["barcode"]),
        *one_hot(states["brand"]),
        *one_hot(states["nutrition"]),
        missing / 4.0,
        1.0,  # intercept
    ])


# --- synthetic training pairs -------------------------------------------------

def _name_variant(name: str, rng: np.random.Generator) -> str:
    choice = rng.integers(0, 5)
    if choice == 0:
        return name.upper()
    if choice == 1:
        return f"{name} {int(rng.integers(20, 120))} g"
    if choice == 2:
        tokens = name.replace(",", " ").split()
        if len(tokens) > 1:
            tokens.pop(int(rng.integers(0, len(tokens))))
        return " ".join(tokens)
    if choice == 3 and len(name) > 3:
        i = int(rng.integers(0, len(name) - 1))
        return name[:i] + name[i + 1] + name[i] + name[i + 2:]
    return name.replace(" ", "-")


def _synth_incoming(food: dict, rng: np.random.Generator) -> dict:
    """A plausible per-serving record for the given canonical food."""
    grams = float(rng.integers(20, 150))
    jitter = 1.0 + rng.uniform(-0.04, 0.04)
    record = {
        "name": _name_variant(food["name"], rng),
        "brand": food.get("brand") if rng.random() > 0.4 else None,
        "barcode": food.get("barcode") if rng.random() > 0.5 else None,
        "serving_quantity": grams,
        "serving_unit": str(rng.choice(["g", "grams"])),
    }
    for k in ("calories", "protein", "carbs", "fat"):
        per100 = food.get(f"{k}_per_100g")
        if per100 is not None:
            record[k] = per100 * grams / 100.0 * jitter
    return record


def _training_data(catalogue: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(TRAINING_SEED)
    rows, labels = [], []

    for food in catalogue:
        for _ in range(POSITIVES_PER_FOOD):
            rows.append(_features(_synth_incoming(food, rng), food))
            labels.append(1.0)

    for a in catalogue:
        for b in catalogue:
            if a["id"] == b["id"]:
                continue
            for _ in range(NEGATIVES_PER_PAIR):
                rows.append(_features(_synth_incoming(a, rng), b))
                labels.append(0.0)
            # w-300 style: record looks like A but carries B's barcode
            for _ in range(HARD_NEGATIVES_PER_PAIR):
                swapped = _synth_incoming(a, rng) | {"barcode": b.get("barcode")}
                rows.append(_features(swapped, b))
                labels.append(0.0)

    return np.array(rows), np.array(labels)


# --- numpy logistic regression ------------------------------------------------

class _LogisticModel:
    def __init__(self, weights: np.ndarray):
        self.weights = weights

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray) -> "_LogisticModel":
        w = np.zeros(x.shape[1])
        for _ in range(EPOCHS):
            p = _sigmoid(x @ w)
            grad = x.T @ (p - y) / len(y) + L2 * w
            w -= LEARNING_RATE * grad
        return cls(w)

    def predict_proba(self, features: np.ndarray) -> float:
        return float(_sigmoid(features @ self.weights))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _catalogue_model(catalogue: list[dict]) -> _LogisticModel | None:
    key = tuple(sorted(f["id"] for f in catalogue))
    if key not in _MODEL_CACHE:
        if len(catalogue) < 2:
            _MODEL_CACHE[key] = None  # no negative pairs to learn from
        else:
            _MODEL_CACHE[key] = _LogisticModel.fit(*_training_data(catalogue))
    return _MODEL_CACHE[key]


def match_probability(incoming: dict, food: dict, catalogue: list[dict]) -> float | None:
    model = _catalogue_model(catalogue)
    if model is None:
        return None
    return model.predict_proba(_features(incoming, food))
