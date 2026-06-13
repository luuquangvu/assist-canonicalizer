# Assist Canonicalizer Evaluation

## Overall

- `hassil`: 31.0% intent/slot, 31.0% canonical, 8.3% mismatch, 60.7% fallback
- `lexical`: 83.7% intent/slot, 69.9% canonical, 8.9% mismatch, 7.4% fallback

## DE

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 30525 (build latency: 929.8ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    87 |       33.3% |     33.3% |    11.5% |    55.2% |   43.1 |
| `lexical` |    87 |       89.7% |     70.1% |     3.4% |     6.9% |   50.4 |

## EN

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 20576 (build latency: 578.6ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   111 |       30.6% |     30.6% |    13.5% |    55.9% |    8.8 |
| `lexical` |   111 |       88.3% |     67.6% |     6.3% |     5.4% |   45.1 |

## FR

- Builtin intents: 34
- Candidate intents: 34
- Dataset intents: 34
- Candidates: 25549 (build latency: 653.4ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    85 |       35.3% |     35.3% |     7.1% |    57.6% |   10.5 |
| `lexical` |    85 |       85.9% |     65.9% |     5.9% |     8.2% |   63.7 |

## NL

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 25328 (build latency: 674.3ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   105 |       34.3% |     34.3% |     4.8% |    61.0% |   53.1 |
| `lexical` |   105 |       74.3% |     78.1% |    15.2% |    10.5% |   49.3 |

## VI

- Builtin intents: 17
- Candidate intents: 17
- Dataset intents: 17
- Candidates: 6446 (build latency: 278.8ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    83 |       20.5% |     20.5% |     3.6% |    75.9% |   14.8 |
| `lexical` |    83 |       80.7% |     66.3% |    13.3% |     6.0% |   60.4 |
