# Assist Canonicalizer Evaluation

**Report schema:** v1

**Dependency versions:** homeassistant=2026.7.2, home-assistant-intents=2026.6.24

## Overall

- `hassil`: 48.3% intent/slot, 48.3% top-1, 48.3% canonical, 0.0% mismatch, 51.7% fallback
- `lexical`: 92.3% intent/slot, 94.2% top-1, 62.0% canonical, 1.4% mismatch, 6.3% fallback

## DE

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 89127 (build latency: 2846.2ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Top-1 | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | ----: | --------: | -------: | -------: | -----: |
| `hassil`  |   112 |       50.9% | 50.9% |     50.9% |     0.0% |    49.1% |   46.9 |
| `lexical` |   112 |       94.6% | 96.4% |     54.5% |     0.9% |     4.5% |   79.8 |

## EN

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 50725 (build latency: 1351.7ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Top-1 | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | ----: | --------: | -------: | -------: | -----: |
| `hassil`  |   121 |       56.2% | 56.2% |     56.2% |     0.0% |    43.8% |    5.6 |
| `lexical` |   121 |       91.7% | 91.7% |     62.0% |     2.5% |     5.8% |   78.1 |

## FR

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 59283 (build latency: 1873.5ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Top-1 | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | ----: | --------: | -------: | -------: | -----: |
| `hassil`  |   109 |       51.4% | 51.4% |     51.4% |     0.0% |    48.6% |    5.0 |
| `lexical` |   109 |       93.6% | 95.4% |     55.0% |     0.9% |     5.5% |   95.5 |

## NL

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 68891 (build latency: 1915.0ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Top-1 | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | ----: | --------: | -------: | -------: | -----: |
| `hassil`  |   122 |       48.4% | 48.4% |     48.4% |     0.0% |    51.6% |   25.8 |
| `lexical` |   122 |       89.3% | 93.4% |     73.8% |     1.6% |     9.0% |   83.1 |

## VI

- Builtin intents: 17
- Candidate intents: 17
- Dataset intents: 17
- Candidates: 17752 (build latency: 635.1ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Top-1 | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | ----: | --------: | -------: | -------: | -----: |
| `hassil`  |    91 |       30.8% | 30.8% |     30.8% |     0.0% |    69.2% |   13.0 |
| `lexical` |    91 |       92.3% | 94.5% |     63.7% |     1.1% |     6.6% |   50.1 |
