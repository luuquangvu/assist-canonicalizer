# Managed Home Assistant Benchmark Environment

This directory contains the tracked Home Assistant environment used by the repository's managed-live benchmark. The runner creates a fresh instance from these files for each run.

Small environment differences can skew voice benchmark results. Examples include a stale registry, renamed entity, or dependency update. This tracked environment records the home model, corpus, source checkout, and runtime versions so maintainers can reproduce and explain a result.

This environment is not intended for interactive development and does not preserve manual configuration. Use a normal, user-managed Home Assistant instance for exploratory testing or integration development.

Use the managed-live runner when a change needs end-to-end accuracy or latency evidence through a real Home Assistant process. Use `--list-cases` to inspect corpus selection without starting Home Assistant. Use `tools/benchmark_offline.py` for focused algorithm diagnostics, not managed-live accuracy evidence.

## Fixture contract

The benchmark uses the predefined `medium_home_v1` fixture:

- **3 floors**: Ground Floor, Upper Floor, and Exterior
- **12 areas**: Distributed across the floors
- **60 entities**: Exposed to Assist across 19 domains exercised by the benchmark (`alarm_control_panel`, `binary_sensor`, `button`, `climate`, `cover`, `fan`, `humidifier`, `lawn_mower`, `light`, `lock`, `media_player`, `sensor`, `siren`, `switch`, `todo`, `vacuum`, `valve`, `water_heater`, and `weather`)

Names and aliases cover English, German, French, Dutch, and Vietnamese. The fixture also includes Simplified and Traditional Chinese aliases used by compatibility checks.

The tracked fixture fingerprint is:

```text
f63468a726b289243ca9ff6b0e387f8e470bc5c0c5239b051e2846cd93cbf9e8
```

Tracked benchmark inputs include:

- `configuration.yaml`: Configures locale and unit settings, enabled components, the loopback HTTP endpoint, and fixture activation.
- `custom_components/assist_canonicalizer_benchmark/fixture.json`: Defines floors, areas, entity identities, names, aliases, and exposure rules.
- `custom_components/assist_canonicalizer_benchmark/__init__.py`: Provisions and verifies the fixture, prepares stateful cases, records traces, and publishes readiness metadata.
- `tests/real_world/*.json`: Contains the maintained multilingual corpus, including canonical controls and curated intent, slot, and category labels.

## Benchmark lifecycle

The benchmark runner (`tools/benchmark.py`) manages the following lifecycle:

1. **Input and dependency verification**: Loads and fingerprints the tracked inputs, then verifies that installed packages satisfy the Home Assistant component manifests.
2. **Temporary configuration**: Confirms that the loopback port is available, creates a temporary directory under `scratch/`, and links the tracked configuration, active Assist Canonicalizer source, and fixture component.
3. **Startup**: Validates the generated Home Assistant configuration and starts the Home Assistant process.
4. **Onboarding**: Creates an ephemeral owner, waits for the initial fixture, and adds the shopping list integration.
5. **Fixture and integration setup**: Reapplies and verifies the fixture, adds Assist Canonicalizer, and prepares the required language indexes. Verification checks the fixture ID, entity counts, domain distribution, and fingerprint.
6. **Execution and evaluation**: Runs each query first through the built-in Assist pipeline and then through Assist Canonicalizer, resetting prerequisite state before each side of the comparison. Responses, traces, resolved entity IDs, and slots are evaluated against the live canonical control.
7. **Cleanup**: Stops Home Assistant and removes the temporary configuration, generated state, database, and ephemeral credentials.

> [!NOTE]
> Home Assistant state is not reused between runs. The offline evaluator (`tools/benchmark_offline.py`) remains useful for diagnostics and focused profiling, but its results are not managed-live accuracy evidence.

## Running the benchmark

From the repository root, prepare the environment and run the complete suite:

```bash
uv sync --all-groups
uv run tools/benchmark.py
```

By default, the runner executes every query in `tests/real_world/`, runs the managed compatibility checks, and writes:

- `scratch/benchmark/managed_live_report.json`
- `scratch/benchmark/managed_live_report.md`

Common selection options include:

- `--languages en,de` to select languages
- `--categories exact_match` to select categories
- `--case-limit 50` to select a deterministic prefix for investigation
- `--list-cases` to validate and list selected cases without starting Home Assistant

### Comparison and baseline matching

The generated report distinguishes successful execution from semantic correctness. Its corpus metrics include:

- **Direct HassIL accuracy**: Accuracy through Home Assistant's built-in conversation agent
- **Assist Canonicalizer accuracy**: Production accuracy after HassIL-first protection, followed by Assist Canonicalizer only when HassIL does not match
- **Assist Canonicalizer mismatch and fallback**: The remaining mutually exclusive production outcomes; together with accuracy they partition the corpus
- **Direct canonicalizer outcomes**: Unprotected accuracy, mismatch, and fallback retained under explicit `direct_canonicalizer_*` fields
- **Uplift**: Difference between Assist Canonicalizer and direct HassIL accuracy in percentage points
- **Recovered cases**: Queries that failed through direct HassIL but passed through Assist Canonicalizer
- **Regressions prevented**: Queries that passed through direct HassIL and therefore never expose a weaker direct canonicalizer result to the user

To preserve both reports while comparing a local change:

```bash
uv run tools/benchmark.py \
  --output-json scratch/benchmark/before.json \
  --output-markdown scratch/benchmark/before.md
uv run tools/benchmark.py \
  --baseline scratch/benchmark/before.json \
  --output-json scratch/benchmark/after.json \
  --output-markdown scratch/benchmark/after.md \
  --fail-on-regression
```

The comparison reports decreases in passed cases, individual case regressions, and p95 latency increases above `--max-p95-regression-pct`, which defaults to 10%. By default, the reports must use the same Python and Home Assistant versions, dependencies, fixture, corpus, configuration, and run settings.

For an intentional Python, Home Assistant, or dependency upgrade, add `--allow-homeassistant-upgrade`. Upgrade comparisons still require the same integration source, fixture, corpus, configuration, and run settings so the environment change remains isolated.

> [!IMPORTANT]
> Accuracy results describe the tracked fixture and corpus. Latency depends on the host, so compare performance reports from the same machine under similar conditions.

## Reproducibility and isolation

The runner controls the main inputs that affect comparisons:

- **Dependency resolution**: `uv.lock` pins package versions. `pyproject.toml` defines the minimum Python version, while each report records the actual Python, Home Assistant, and package versions.
- **Fresh state**: Every run starts without prior `.storage` data, registries, database contents, or pipeline configuration.
- **Active source checkout**: The integration source is linked directly from the current checkout.
- **Loopback endpoint**: Home Assistant binds to `127.0.0.1:8123`. The runner checks that the port is available before starting Home Assistant.
- **Fixture verification**: The run stops if the floor, area, exposed-entity, domain, or fingerprint contract differs.
- **State preparation**: Stateful prerequisites, such as timers, to-do items, shopping-list entries, and media-player state, are prepared for each paired request.
- **Live oracles**: Outcomes are evaluated using the running Home Assistant intent handlers rather than an offline reimplementation.

## Maintaining the benchmark contract

Fixture and corpus changes alter what the benchmark measures. When changing either:

- Explain why a floor, area, entity, alias, language case, or expected control needs to change.
- Keep fixture counts, domain distributions, corpus labels, and their tests in sync.
- Update the documented fixture fingerprint and establish a new baseline. Reports with different fixtures or corpora are not directly comparable.
- Use upgrade comparison mode for Python, Home Assistant, or package changes while keeping the integration source unchanged.

These steps keep results reproducible and make it clear when two reports can be compared.
