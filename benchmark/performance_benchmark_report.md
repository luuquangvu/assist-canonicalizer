# Assist Canonicalizer Evaluation

## Overall

- `hassil`: 31.0% intent/slot, 31.0% canonical, 8.3% mismatch, 60.7% fallback
- `lexical`: 81.7% intent/slot, 69.2% canonical, 8.3% mismatch, 10.0% fallback

## DE

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 30525 (build latency: 976.3ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    87 |       33.3% |     33.3% |    11.5% |    55.2% |   64.0 |
| `lexical` |    87 |       88.5% |     70.1% |     3.4% |     8.0% |   54.0 |

## EN

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 20576 (build latency: 551.0ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   111 |       30.6% |     30.6% |    13.5% |    55.9% |   10.0 |
| `lexical` |   111 |       86.5% |     68.5% |     6.3% |     7.2% |   40.1 |

## FR

- Builtin intents: 34
- Candidate intents: 34
- Dataset intents: 34
- Candidates: 25549 (build latency: 558.6ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    85 |       35.3% |     35.3% |     7.1% |    57.6% |   10.5 |
| `lexical` |    85 |       82.4% |     64.7% |     2.4% |    15.3% |   56.3 |

## NL

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 25328 (build latency: 745.9ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   105 |       34.3% |     34.3% |     4.8% |    61.0% |   68.0 |
| `lexical` |   105 |       73.3% |     78.1% |    15.2% |    11.4% |   46.4 |

## VI

- Builtin intents: 17
- Candidate intents: 17
- Dataset intents: 17
- Candidates: 6446 (build latency: 190.3ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    83 |       20.5% |     20.5% |     3.6% |    75.9% |   14.9 |
| `lexical` |    83 |       78.3% |     62.7% |    13.3% |     8.4% |   44.6 |
