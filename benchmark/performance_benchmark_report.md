# Assist Canonicalizer Evaluation

**Report schema:** v1

**Dependency versions:** homeassistant=2026.7.1, home-assistant-intents=2026.6.24

## Overall

- `hassil`: 48.3% intent/slot, 48.3% top-1, 48.3% canonical, 0.0% mismatch, 51.7% fallback
- `lexical`: 91.7% intent/slot, 93.7% top-1, 61.6% canonical, 2.2% mismatch, 6.1% fallback

## DE

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 89127 (build latency: 2479.7ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Top-1 | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | ----: | --------: | -------: | -------: | -----: |
| `hassil`  |   112 |       50.9% | 50.9% |     50.9% |     0.0% |    49.1% |   47.6 |
| `lexical` |   112 |       94.6% | 96.4% |     54.5% |     0.9% |     4.5% |  111.4 |

## EN

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 50725 (build latency: 1216.8ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Top-1 | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | ----: | --------: | -------: | -------: | -----: |
| `hassil`  |   121 |       56.2% | 56.2% |     56.2% |     0.0% |    43.8% |    4.8 |
| `lexical` |   121 |       90.1% | 90.1% |     60.3% |     4.1% |     5.8% |   96.0 |

## FR

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 59283 (build latency: 1590.7ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Top-1 | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | ----: | --------: | -------: | -------: | -----: |
| `hassil`  |   109 |       51.4% | 51.4% |     51.4% |     0.0% |    48.6% |    4.5 |
| `lexical` |   109 |       92.7% | 94.5% |     55.0% |     1.8% |     5.5% |  133.7 |

## NL

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 68891 (build latency: 1990.9ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Top-1 | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | ----: | --------: | -------: | -------: | -----: |
| `hassil`  |   122 |       48.4% | 48.4% |     48.4% |     0.0% |    51.6% |   26.6 |
| `lexical` |   122 |       89.3% | 93.4% |     73.8% |     2.5% |     8.2% |   98.2 |

## VI

- Builtin intents: 17
- Candidate intents: 17
- Dataset intents: 17
- Candidates: 17752 (build latency: 587.4ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Top-1 | Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | ----: | --------: | -------: | -------: | -----: |
| `hassil`  |    91 |       30.8% | 30.8% |     30.8% |     0.0% |    69.2% |   11.4 |
| `lexical` |    91 |       92.3% | 94.5% |     63.7% |     1.1% |     6.6% |   60.7 |
