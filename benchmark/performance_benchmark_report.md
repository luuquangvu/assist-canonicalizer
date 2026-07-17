# Assist Canonicalizer Evaluation

**Report schema:** v1

**Dependency versions:** homeassistant=2026.7.2, home-assistant-intents=2026.6.24

## Overall

- `hassil`: 47.7% intent/slot, 47.7% exact canonical, 0.0% mismatch, 52.3% fallback
- `lexical`: 92.8% intent/slot, 69.4% exact canonical, 1.3% mismatch, 5.8% fallback

## Production-Flow Component Top-1: ALL LANGUAGES

Cohort: 599/599 evaluated | production_fallbacks=35 | excluded drift=0

Exact Canonical = selected command text exactly matches the expected canonical text; Intent/Slot = selected intent and slots match the expected semantics.

| Component       | Evaluated | Exact Canonical |     Intent/Slot |
| :-------------- | --------: | --------------: | --------------: |
| `rapidfuzz`     |       599 | 384/599 (64.1%) | 541/599 (90.3%) |
| `char_ngram`    |       599 | 415/599 (69.3%) | 539/599 (90.0%) |
| `bm25`          |       599 | 394/599 (65.8%) | 542/599 (90.5%) |
| `intent_action` |       599 | 396/599 (66.1%) | 532/599 (88.8%) |
| `final`         |       599 | 416/599 (69.4%) | 556/599 (92.8%) |

## DE

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 89127 (build latency: 3580.5ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Exact Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------------: | -------: | -------: | -----: |
| `hassil`  |   122 |       48.4% |           48.4% |     0.0% |    51.6% |   51.4 |
| `lexical` |   122 |       94.3% |           66.4% |     0.8% |     4.9% |  120.2 |

## EN

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 50725 (build latency: 1726.9ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Exact Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------------: | -------: | -------: | -----: |
| `hassil`  |   129 |       52.7% |           52.7% |     0.0% |    47.3% |    5.2 |
| `lexical` |   129 |       92.2% |           70.5% |     2.3% |     5.4% |   76.7 |

## FR

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 59283 (build latency: 2162.7ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Exact Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------------: | -------: | -------: | -----: |
| `hassil`  |   119 |       49.6% |           49.6% |     0.0% |    50.4% |    4.7 |
| `lexical` |   119 |       93.3% |           63.9% |     1.7% |     5.0% |   92.9 |

## NL

- Builtin intents: 41
- Candidate intents: 41
- Dataset intents: 41
- Candidates: 68891 (build latency: 2058.5ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Exact Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------------: | -------: | -------: | -----: |
| `hassil`  |   129 |       48.8% |           48.8% |     0.0% |    51.2% |   26.0 |
| `lexical` |   129 |       91.5% |           77.5% |     0.8% |     7.8% |   98.0 |

## VI

- Builtin intents: 17
- Candidate intents: 17
- Dataset intents: 17
- Candidates: 17752 (build latency: 727.8ms)
- Missing candidate intents: 0
- Untested candidate intents: 0

| Mode      | Total | Intent/Slot | Exact Canonical | Mismatch | Fallback | Avg ms |
| :-------- | ----: | ----------: | --------------: | -------: | -------: | -----: |
| `hassil`  |   100 |       37.0% |           37.0% |     0.0% |    63.0% |   12.6 |
| `lexical` |   100 |       93.0% |           68.0% |     1.0% |     6.0% |   67.2 |
