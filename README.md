# Hexis CTO Live Coding Exercise

## Validated Food Catalogue Matcher

Hexis receives food records from manufacturers, commercial and public databases, and users. Records can be incomplete, duplicated, differently formatted or contradictory.

Implement a small component that decides whether an incoming record should be automatically matched to a canonical food, sent for human review, or treated as a new food. Incorrectly merging two different foods is more damaging than sending a valid match for review.

## Output

Return a result with:

- `canonical_food_id`: string or null
- `decision`: `AUTO_ACCEPT`, `REVIEW`, or `NEW`
- `confidence`: number or null
- `reasons`: list of human-readable strings

## Core requirements

1. Normalise records so comparable values use a consistent representation.
2. Evaluate each incoming record against the supplied canonical catalogue and identify the best candidate.
3. Treat a matching barcode as strong evidence, but do not ignore conflicting evidence.
4. Incoming nutrition values are per serving. Convert them to per 100 g when a mass serving is available.
5. Treat missing evidence differently from contradictory evidence.
6. Route ambiguous or conflicting records to `REVIEW`.
7. Return human-readable reasons for every decision.
8. Add automated tests for the behaviours you consider most important.

You may choose the matching strategy, thresholds and data structures. State assumptions. You do not need to implement a database, user interface, live web scraper or machine-learning model.

The interviewers may introduce one changed requirement during the session.

---

## Solution

- `matcher.py`: normalization, evidence scoring, and the decision rules.
- `ml.py`: learned layer (numpy logistic regression + optional embedding similarity), strictly subordinate to the deterministic guardrails.
- `test_matcher.py` / `test_ml.py`: behavioral tests, including the safety invariants.
- `docs/ARCHITECTURE.md`: where this component sits in the wider platform.

### Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q          # tests
.venv/bin/python matcher.py  # decision map for the supplied fixtures
```

### Approach

Each incoming record is scored against every canonical food on four
independent signals: barcode (exact, digits-only), name (lexical fuzzy
similarity, optionally boosted by embeddings for ranking only), brand,
and nutrition (per-serving converted to per-100 g when the serving is a
mass, then compared within tolerance). Every signal is tri-state:
**match / missing / conflict** — missing evidence contributes nothing,
conflicting evidence is recorded and vetoes auto-acceptance.

Decision rules, in order:

1. Any conflict on the best candidate, or the incoming barcode belonging
   to a *different* canonical food → `REVIEW`.
2. Two candidates scoring within a small margin → ambiguous → `REVIEW`.
3. Barcode match auto-accepts only when **every other comparable signal
   fully agrees**; without a barcode, auto-accept needs a near-exact name
   plus agreeing nutrition.
4. A learned match probability (logistic regression trained on
   catalogue-derived synthetic pairs) supplies the calibrated confidence
   and can downgrade a rule-approved accept — it can veto a merge, never
   force one.
5. Nothing scores above the floor and no barcode matches → `NEW`.

### Assumptions

- Canonical nutrition is per 100 g; incoming nutrition is per serving.
- Mass units convertible to grams: g/gram(s)/gr, kg, oz. Anything else
  (ml, scoops) makes nutrition *missing* evidence, not contradictory.
- Barcodes compare digits-only and exactly; leading zeros preserved.
- Wrong merges cost more than reviews, so all thresholds are biased
  toward `REVIEW`; they live as named constants at the top of
  `matcher.py` so a requirement change is a one-line edit.
