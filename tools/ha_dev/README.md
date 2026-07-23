# Managed Home Assistant Benchmark Environment

This directory defines the repository's single Home Assistant benchmark home.

This setup is not intended as an interactive development installation and does not preserve a manually configured Home Assistant instance. For manual testing or development, please use a standard, user-managed Home Assistant instance.

## Baseline Contract

The benchmark uses a predefined Home Assistant model (`medium_home_v1`) containing:

- **3 Floors**: Ground Floor, Upper Floor, and Exterior
- **12 Areas**: Distributed across the floors
- **60 Entities**: Exposed to Assist across all 19 target domains supported by built-in intent handlers (`alarm_control_panel`, `binary_sensor`, `button`, `climate`, `cover`, `fan`, `humidifier`, `lawn_mower`, `light`, `lock`, `media_player`, `sensor`, `siren`, `switch`, `todo`, `vacuum`, `valve`, `water_heater`, `weather`)

Names and aliases are configured for English, German, French, Dutch, and Vietnamese, with Simplified and Traditional Chinese aliases for compatibility checks.

The reviewed user-model fingerprint is:

```text
f63468a726b289243ca9ff6b0e387f8e470bc5c0c5239b051e2846cd93cbf9e8
```

Tracked benchmark inputs include:

- `configuration.yaml`: Configures locales, units, enabled components, loopback HTTP bindings, and fixture activation.
- `custom_components/assist_canonicalizer_benchmark/fixture.json`: Defines floors, areas, entity identities, names, aliases, and exposure rules.
- `custom_components/assist_canonicalizer_benchmark/__init__.py`: Handles provisioning, verification, live recognition checks, stateful resets, passive tracing, and readiness/fingerprint publication.
- `tests/real_world/*.json`: The authoritative multilingual test corpus, including canonical controls and migration labels.

## Benchmark Lifecycle

The benchmark runner (`tools/benchmark.py`) orchestrates the complete lifecycle for each run:

1. **Setup**: Creates an isolated temporary configuration directory within `scratch/`.
2. **Symlinking**: Links the tracked configuration files, active Assist Canonicalizer source code, and the benchmark fixture component.
3. **Dependency Verification**: Verifies that installed packages satisfy the requirements declared in Home Assistant component manifests, validates configurations, and starts the Home Assistant process.
4. **Onboarding**: Completes onboarding using ephemeral credentials, registers the necessary integrations (shopping list and assist_canonicalizer), and waits for the benchmark fixture to report readiness.
5. **Fixture Verification**: Validates the fixture ID, entity counts, domain distributions, and fingerprint values.
6. **Execution & Evaluation**: Executes every query as a paired test case: running first through the built-in Assist pipeline, resetting state, and then running through Assist Canonicalizer. Results (traces, responses, resolved entity IDs, and slots) are compared against the canonical control.
7. **Cleanup**: Removes all temporary Home Assistant states, credentials, and configuration files.

> [!NOTE]
> There is no persistent Home Assistant state. The runner starts with a completely clean environment for every execution. The offline evaluator (`tools/benchmark_offline.py`) remains available for diagnostics and micro-benchmarks, but its results are non-authoritative.

## Running the Benchmark

To prepare the environment and run the full benchmark suite:

```bash
uv sync --all-groups
uv run tools/benchmark.py
```

By default, the runner executes all queries in `tests/real_world/` and writes the reports to `scratch/benchmark/managed_live_report.json` and `scratch/benchmark/managed_live_report.md`.

You can customize the run using flags:

- Filter by languages: `--languages en,de`
- Filter by categories: `--categories exact_match`
- Limit the case count: `--case-limit 50`
- List and inspect cases without starting Home Assistant: `--list-cases`

### Comparison and Baseline Matching

The generated report distinguishes semantic correctness from successful execution. It tracks:

- **Direct-HassIL Accuracy**: Baseline accuracy of default Assist.
- **Canonicalizer Accuracy**: Accuracy after canonicalization.
- **Uplift**: Percentage-point improvement.
- **Recovered Cases**: Queries that failed on HassIL but succeeded via the canonicalizer.
- **Regressed Cases**: Queries that succeeded on HassIL but failed via the canonicalizer.

To perform a before/after comparison on local changes:

```bash
uv run tools/benchmark.py \
  --output-json scratch/benchmark/before.json
uv run tools/benchmark.py \
  --baseline scratch/benchmark/before.json \
  --fail-on-regression
```

The comparison checks for regressions in functional behavior and p95 latency. It blocks runs that introduce drift in Python versions, Home Assistant, configurations, or case definitions. For intentional upgrades of the core Home Assistant environment, use the `--allow-homeassistant-upgrade` flag.

## Consistency and Environment Isolation

To ensure benchmarks are completely deterministic and isolated:

- **Dependency Resolution**: `uv.lock` locks the versions of Python, Home Assistant, and other dependencies. While `pyproject.toml` leaves runner packages unpinned, the runner resolves constraints directly against installed manifests.
- **Clean Slate**: Each benchmark starts with an empty Home Assistant state (`.storage`, registries, database, and pipeline configurations are fully clean). No configuration is retained between runs.
- **Direct Integration Testing**: The integration source is symlinked directly from the active git checkout to prevent version drift.
- **Port Binding**: Home Assistant binds exclusively to local loopback (`127.0.0.1:8123`). The runner fails immediately if the port is already in use.
- **Fixture Verification**: Setup halts if any discrepancies are found in exposed floors, areas, entity domains, or the fixture fingerprint.
- **State Resets**: Prerequisite states (such as active timers, to-do list items, shopping lists, or media player statuses) are reset before each case query is sent.
- **Runtime Oracles**: The runner evaluates outcomes using the running instance's intent handlers rather than recreating intent parsing or canonicalization rules offline.

Any changes to the fixture layout require updating the baseline fingerprint. When upgrading Home Assistant, use the upgrade comparison mode to verify performance under the new environment while holding integration logic constant.
