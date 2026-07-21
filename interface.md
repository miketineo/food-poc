# Language-neutral interface suggestion

This is only a suggestion. Use a form natural to your chosen language.

```text
match_food(incoming_food, canonical_catalogue) -> MatchResult

MatchResult
  canonical_food_id: string | null
  decision: AUTO_ACCEPT | REVIEW | NEW
  confidence: number | null
  reasons: string[]
```
