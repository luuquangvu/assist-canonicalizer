# Assist Canonicalizer Evaluation

**Dependency versions:** homeassistant=2026.6.4, home-assistant-intents=2026.6.1

## Overall

- `hassil`: 45.9% intent/slot, 45.9% canonical, 0.0% mismatch, 54.1% fallback
- `lexical`: 86.2% intent/slot, 65.2% canonical, 2.3% mismatch, 11.5% fallback

## DE

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 80849 (build latency: 2993.3ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    87 |       50.6% |     50.6% |     0.0% |    49.4% |   44.0 |
| `lexical` |    87 |       86.2% |     66.7% |     2.3% |    11.5% |   74.7 |

## EN

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 41075 (build latency: 1467.8ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   111 |       51.4% |     51.4% |     0.0% |    48.6% |    7.2 |
| `lexical` |   111 |       90.1% |     64.0% |     2.7% |     7.2% |   56.4 |

## FR

- Builtin intents: 34
- Candidate intents: 34
- Dataset intents: 34
- Candidates: 50602 (build latency: 1264.7ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    85 |       49.4% |     49.4% |     0.0% |    50.6% |    7.5 |
| `lexical` |    85 |       88.2% |     60.0% |     1.2% |    10.6% |   88.6 |

## NL

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 49031 (build latency: 1854.2ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   105 |       44.8% |     44.8% |     0.0% |    55.2% |   50.3 |
| `lexical` |   105 |       82.9% |     72.4% |     1.9% |    15.2% |   72.9 |

## VI

- Builtin intents: 17
- Candidate intents: 17
- Dataset intents: 17
- Candidates: 17248 (build latency: 773.5ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |    83 |       31.3% |     31.3% |     0.0% |    68.7% |   13.4 |
| `lexical` |    83 |       83.1% |     61.4% |     3.6% |    13.3% |   53.7 |
