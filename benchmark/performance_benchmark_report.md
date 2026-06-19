# Assist Canonicalizer Evaluation

## Overall

- `hassil`: 34.0% intent/slot, 34.0% canonical, 5.3% mismatch, 60.7% fallback
- `lexical`: 83.7% intent/slot, 69.9% canonical, 8.9% mismatch, 7.4% fallback

## DE

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 30525 (build latency: 641.5ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    87 |       36.8% |     36.8% |     8.0% |    55.2% |   42.3 |
| `lexical` |    87 |       89.7% |     70.1% |     3.4% |     6.9% |   48.7 |

## EN

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 20576 (build latency: 535.3ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   111 |       37.8% |     37.8% |     6.3% |    55.9% |    9.0 |
| `lexical` |   111 |       88.3% |     67.6% |     6.3% |     5.4% |   33.8 |

## FR

- Builtin intents: 34
- Candidate intents: 34
- Dataset intents: 34
- Candidates: 25549 (build latency: 428.4ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    85 |       36.5% |     36.5% |     5.9% |    57.6% |    9.7 |
| `lexical` |    85 |       85.9% |     65.9% |     5.9% |     8.2% |   64.0 |

## NL

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 25328 (build latency: 483.8ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   105 |       34.3% |     34.3% |     4.8% |    61.0% |   52.4 |
| `lexical` |   105 |       74.3% |     78.1% |    15.2% |    10.5% |   45.4 |

## VI

- Builtin intents: 17
- Candidate intents: 17
- Dataset intents: 17
- Candidates: 6446 (build latency: 141.0ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    83 |       22.9% |     22.9% |     1.2% |    75.9% |   11.3 |
| `lexical` |    83 |       80.7% |     66.3% |    13.3% |     6.0% |   38.0 |
