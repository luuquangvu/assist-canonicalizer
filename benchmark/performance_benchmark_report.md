# Assist Canonicalizer Evaluation

**Report schema:** v1

**Dependency versions:** homeassistant=2026.7.2, home-assistant-intents=2026.6.24

## Overall

- `hassil`: 47.7% intent/slot, 47.7% canonical, 0.0% mismatch, 52.3% fallback
- `lexical`: 92.7% intent/slot, 68.3% canonical, 1.5% mismatch, 5.8% fallback

## DE

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 89127 (build latency: 2765.1ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   122 |       48.4% |     48.4% |     0.0% |    51.6% |   46.7 |
| `lexical` |   122 |       95.1% |     61.5% |     0.8% |     4.1% |   96.7 |

## EN

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 50725 (build latency: 1302.4ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   129 |       52.7% |     52.7% |     0.0% |    47.3% |    5.6 |
| `lexical` |   129 |       92.2% |     70.5% |     2.3% |     5.4% |   64.9 |

## FR

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 59283 (build latency: 1868.9ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   119 |       49.6% |     49.6% |     0.0% |    50.4% |    4.7 |
| `lexical` |   119 |       93.3% |     63.9% |     1.7% |     5.0% |   76.6 |

## NL

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 68891 (build latency: 1709.8ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   129 |       48.8% |     48.8% |     0.0% |    51.2% |   28.2 |
| `lexical` |   129 |       89.9% |     76.0% |     1.6% |     8.5% |   89.6 |

## VI

- Builtin intents: 17
- Candidate intents: 17
- Dataset intents: 17
- Candidates: 17752 (build latency: 611.6ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   100 |       37.0% |     37.0% |     0.0% |    63.0% |   14.2 |
| `lexical` |   100 |       93.0% |     69.0% |     1.0% |     6.0% |   55.7 |
