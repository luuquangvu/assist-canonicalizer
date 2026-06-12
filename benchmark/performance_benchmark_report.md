# Assist Canonicalizer Evaluation

## Overall

- `hassil`: 31.1% intent/slot, 31.1% canonical, 8.3% mismatch, 60.6% fallback
- `lexical`: 81.7% intent/slot, 69.1% canonical, 8.3% mismatch, 10.0% fallback

## DE

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 30525 (build latency: 884.2ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    87 |       33.3% |     33.3% |    11.5% |    55.2% |   63.5 |
| `lexical` |    87 |       88.5% |     70.1% |     3.4% |     8.0% |   53.2 |

## EN

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 20576 (build latency: 577.4ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   111 |       30.6% |     30.6% |    13.5% |    55.9% |   10.7 |
| `lexical` |   111 |       86.5% |     68.5% |     6.3% |     7.2% |   41.9 |

## FR

- Builtin intents: 34
- Candidate intents: 34
- Dataset intents: 34
- Candidates: 25549 (build latency: 614.1ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    85 |       35.3% |     35.3% |     7.1% |    57.6% |   11.5 |
| `lexical` |    85 |       82.4% |     64.7% |     2.4% |    15.3% |   57.8 |

## NL

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 25328 (build latency: 850.2ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   105 |       34.3% |     34.3% |     4.8% |    61.0% |   75.1 |
| `lexical` |   105 |       73.3% |     78.1% |    15.2% |    11.4% |   46.4 |

## VI

- Builtin intents: 17
- Candidate intents: 17
- Dataset intents: 16
- Candidates: 6446 (build latency: 192.6ms)
- Missing candidate intents: 0
- Untested candidate intents: 1

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    82 |       20.7% |     20.7% |     3.7% |    75.6% |   16.3 |
| `lexical` |    82 |       78.0% |     62.2% |    13.4% |     8.5% |   41.0 |
