# Assist Canonicalizer Evaluation

## Overall

- `hassil`: 31.0% intent/slot, 31.0% canonical, 8.3% mismatch, 60.7% fallback
- `lexical`: 83.7% intent/slot, 69.9% canonical, 8.9% mismatch, 7.4% fallback

## DE

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 30525 (build latency: 932.6ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    87 |       33.3% |     33.3% |    11.5% |    55.2% |   44.2 |
| `lexical` |    87 |       89.7% |     70.1% |     3.4% |     6.9% |   47.9 |

## EN

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 20576 (build latency: 579.3ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   111 |       30.6% |     30.6% |    13.5% |    55.9% |    8.7 |
| `lexical` |   111 |       88.3% |     67.6% |     6.3% |     5.4% |   43.5 |

## FR

- Builtin intents: 34
- Candidate intents: 34
- Dataset intents: 34
- Candidates: 25549 (build latency: 657.6ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    85 |       35.3% |     35.3% |     7.1% |    57.6% |    9.8 |
| `lexical` |    85 |       85.9% |     65.9% |     5.9% |     8.2% |   58.4 |

## NL

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 25328 (build latency: 675.8ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   105 |       34.3% |     34.3% |     4.8% |    61.0% |   54.4 |
| `lexical` |   105 |       74.3% |     78.1% |    15.2% |    10.5% |   41.8 |

## VI

- Builtin intents: 17
- Candidate intents: 17
- Dataset intents: 17
- Candidates: 6446 (build latency: 280.5ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    83 |       20.5% |     20.5% |     3.6% |    75.9% |   11.8 |
| `lexical` |    83 |       80.7% |     66.3% |    13.3% |     6.0% |   62.5 |
