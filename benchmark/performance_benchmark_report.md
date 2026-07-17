# Assist Canonicalizer Evaluation

**Report schema:** v1

**Dependency versions:** homeassistant=2026.7.2, home-assistant-intents=2026.6.24

## Overall

- `hassil`: 47.7% intent/slot, 47.7% canonical, 0.0% mismatch, 52.3% fallback
- `lexical`: 92.8% intent/slot, 69.6% canonical, 1.3% mismatch, 5.8% fallback

## DE

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 89127 (build latency: 3447.1ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   122 |       48.4% |     48.4% |     0.0% |    51.6% |   46.7 |
| `lexical` |   122 |       94.3% |     66.4% |     0.8% |     4.9% |  106.6 |

## EN

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 50725 (build latency: 1494.2ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   129 |       52.7% |     52.7% |     0.0% |    47.3% |    5.3 |
| `lexical` |   129 |       92.2% |     70.5% |     2.3% |     5.4% |   76.6 |

## FR

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 59283 (build latency: 2607.9ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   119 |       49.6% |     49.6% |     0.0% |    50.4% |    5.9 |
| `lexical` |   119 |       93.3% |     63.9% |     1.7% |     5.0% |  100.8 |

## NL

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 68891 (build latency: 2136.6ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   129 |       48.8% |     48.8% |     0.0% |    51.2% |   27.7 |
| `lexical` |   129 |       91.5% |     77.5% |     0.8% |     7.8% |   98.7 |

## VI

- Builtin intents: 17
- Candidate intents: 17
- Dataset intents: 17
- Candidates: 17752 (build latency: 793.5ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------: | -------: | -------: | -----: |
| `hassil`  |   100 |       37.0% |     37.0% |     0.0% |    63.0% |   12.8 |
| `lexical` |   100 |       93.0% |     69.0% |     1.0% |     6.0% |   63.6 |
