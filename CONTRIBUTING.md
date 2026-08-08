# Contributing to Assist Canonicalizer

Thanks for contributing to Assist Canonicalizer, a Home Assistant custom integration. Please search existing [issues](https://github.com/luuquangvu/assist-canonicalizer/issues) before starting substantial work.

## Development setup

Use a POSIX environment; Linux, or WSL are recommended. The project requires Python 3.14.2 or newer, [uv](https://docs.astral.sh/uv/), and Node.js/npm.

```bash
git clone https://github.com/luuquangvu/assist-canonicalizer.git
cd assist-canonicalizer
uv sync --all-groups
npm ci
```

Use `uv run` for Python commands. Keep `uv.lock` synchronized with `pyproject.toml`, and keep `package-lock.json` synchronized with `package.json`.

Manage Python dependencies exclusively with uv; do not edit `uv.lock` manually.

## Making changes

- Start from an up-to-date `main` branch and keep each pull request focused.
- Read the affected code and tests before editing.
- Add regression tests for behavior changes.
- Update services, translations, documentation, and metadata when applicable.
- Do not change the integration version in ordinary pull requests.
- Review the complete diff before requesting review.

## Tests and validation

Run a focused test while developing, for example:

```bash
uv run pytest tests/test_conversation.py
```

Run the full local gate before submitting:

```bash
uv run tools/validate.py
```

Validation checks dependency alignment, Ruff, Ty, Pyright, Interrogate, Prettier, and the full pytest suite. It passes only when the output contains `VALIDATION_SUCCESS`; Ruff and Prettier may modify files, so review the diff afterward.

Run the compatibility matrix when changing Home Assistant API usage, compatibility code, dependencies, or `tools/compatibility_matrix.json`:

```bash
uv run tools/validate_compatibility.py
```

For user-facing changes, keep `strings.json` and the translation files aligned. Translation tests enforce their structure and key order.

Run the authoritative managed-live benchmark when changing recognition behavior, benchmark inputs, or performance-sensitive code:

```bash
uv run tools/benchmark.py
```

`tools/benchmark_offline.py` is useful for lexical diagnostics and profiling, but is not managed-live accuracy evidence. See [`tools/ha_dev/README.md`](tools/ha_dev/README.md) before changing the tracked benchmark fixture or corpus.

## AI-assisted contributions

AI coding agents may be used. The author remains primarily responsible for every submitted change, including code produced with AI assistance. The author must understand the change, verify it against the existing code and tests, review the full diff, check for security and compatibility issues, and run the required validation. Do not submit generated code that has not been reviewed and tested.

## Pull requests

Describe what changed, why it changed, affected Home Assistant versions or languages, and how it was tested. Include benchmark results when recognition accuracy or latency could change, along with relevant compatibility, persistence, translation, or migration considerations.

CI runs static checks, tests, a managed-live benchmark, a Home Assistant compatibility matrix, Prettier, Hassfest, and HACS validation as applicable.

## Reporting issues

Use the [bug report](https://github.com/luuquangvu/assist-canonicalizer/issues/new?template=bug_report.yaml), [feature request](https://github.com/luuquangvu/assist-canonicalizer/issues/new?template=feature_request.yaml), or [other issue](https://github.com/luuquangvu/assist-canonicalizer/issues/new?template=other.yaml) template. Include your Home Assistant and integration versions, language, exact input, expected result, actual result, and any relevant diagnostics. Be sure to remove any private household details before posting.

## License

By contributing, you agree that your contributions are distributed under the project's [MIT License](LICENSE).
