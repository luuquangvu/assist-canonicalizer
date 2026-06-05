# Assist Canonicalizer Evaluation

## Overall

- `hassil`: 29.8% intent/slot, 29.8% canonical, 9.6% mismatch, 60.6% fallback
- `lexical`: 80.0% intent/slot, 68.1% canonical, 8.3% mismatch, 11.7% fallback

## DE

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 30525 (build latency: 672.1ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    87 |       32.2% |     32.2% |    12.6% |    55.2% |   51.2 |
| `lexical` |    87 |       88.5% |     70.1% |     3.4% |     8.0% |  108.2 |

## EN

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 20576 (build latency: 431.3ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   111 |       28.8% |     28.8% |    15.3% |    55.9% |    7.3 |
| `lexical` |   111 |       86.5% |     68.5% |     6.3% |     7.2% |   87.5 |

## FR

- Builtin intents: 34
- Candidate intents: 34
- Dataset intents: 34
- Candidates: 25549 (build latency: 460.7ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    85 |       34.1% |     34.1% |     8.2% |    57.6% |    9.4 |
| `lexical` |    85 |       81.2% |     63.5% |     2.4% |    16.5% |   94.5 |

## NL

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 25328 (build latency: 668.3ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   105 |       34.3% |     34.3% |     4.8% |    61.0% |   68.5 |
| `lexical` |   105 |       70.5% |     77.1% |    15.2% |    14.3% |  110.4 |

## VI

- Builtin intents: 17
- Candidate intents: 17
- Dataset intents: 17
- Candidates: 6446 (build latency: 156.6ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    82 |       18.3% |     18.3% |     6.1% |    75.6% |   17.3 |
| `lexical` |    82 |       73.2% |     58.5% |    13.4% |    13.4% |   63.6 |
