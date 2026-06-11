# Assist Canonicalizer Evaluation

## Overall

- `hassil`: 29.8% intent/slot, 29.8% canonical, 9.6% mismatch, 60.6% fallback
- `lexical`: 80.9% intent/slot, 68.3% canonical, 8.3% mismatch, 10.9% fallback

## DE

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 30525 (build latency: 659.7ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    87 |       32.2% |     32.2% |    12.6% |    55.2% |   63.3 |
| `lexical` |    87 |       88.5% |     70.1% |     3.4% |     8.0% |  103.5 |

## EN

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 20576 (build latency: 436.9ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   111 |       28.8% |     28.8% |    15.3% |    55.9% |    7.3 |
| `lexical` |   111 |       86.5% |     68.5% |     6.3% |     7.2% |   84.2 |

## FR

- Builtin intents: 34
- Candidate intents: 34
- Dataset intents: 34
- Candidates: 25549 (build latency: 432.0ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    85 |       34.1% |     34.1% |     8.2% |    57.6% |    7.5 |
| `lexical` |    85 |       81.2% |     63.5% |     2.4% |    16.5% |   80.9 |

## NL

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 25328 (build latency: 573.2ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   105 |       34.3% |     34.3% |     4.8% |    61.0% |   66.6 |
| `lexical` |   105 |       73.3% |     78.1% |    15.2% |    11.4% |   93.7 |

## VI

- Builtin intents: 17
- Candidate intents: 17
- Dataset intents: 17
- Candidates: 6446 (build latency: 134.9ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| --------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    82 |       18.3% |     18.3% |     6.1% |    75.6% |   14.4 |
| `lexical` |    82 |       74.4% |     58.5% |    13.4% |    12.2% |   58.3 |
