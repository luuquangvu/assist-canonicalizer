"""Dynamic wildcard rehydration and stem alignment logic."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from functools import lru_cache
from itertools import pairwise
from math import ceil
from typing import NamedTuple

from homeassistant.util.json import JsonObjectType, JsonValueType
from rapidfuzz import fuzz

from .candidate import Candidate
from .const import (
    MAX_TOKEN_LENGTH_RATIO,
    WILDCARD_LITERAL_COVERAGE_THRESHOLD,
    WILDCARD_STEM_ALIGNMENT_THRESHOLD,
)
from .normalization import normalize_text
from .utils import wildcard_slot_names

_TRAILING_DIGIT_RE = re.compile(r"\d\Z")


class WildcardVariantAnalysis(NamedTuple):
    """Literal-token coverage requirements for one wildcard variant."""

    literal_tokens: frozenset[str]
    required_match_count: int


@lru_cache(maxsize=512)
def _raw_cached_fuzz_ratio(s1: str, s2: str) -> float:
    """Return RapidFuzz's ratio with LRU caching."""
    return fuzz.ratio(s1, s2)


def _cached_fuzz_ratio(s1: str, s2: str) -> float:
    """Return RapidFuzz's ratio with LRU caching.

    The argument order is normalized to maximize cache hit rate.
    """
    return _raw_cached_fuzz_ratio(s1, s2) if s1 <= s2 else _raw_cached_fuzz_ratio(s2, s1)


def _normalize_and_split(text: str) -> tuple[str, tuple[str, ...]]:
    """Normalize text and return both the normalized text and its tokens tuple."""
    norm = normalize_text(text)
    return norm, tuple(norm.split())


def _contextual_norm_token_lists(
    original_tokens: tuple[str, ...],
) -> tuple[tuple[str, ...], ...]:
    """Return each original token's normalized tokens, honoring left context.

    Whole-string normalization preserves unit suffixes through a
    digit lookbehind that may cross a whitespace boundary ("50 %" keeps
    the "%" token), while normalizing a token in isolation drops it. All
    other placeholder patterns require adjacency, so replaying a
    digit-plus-space left context reproduces the whole-string reading and
    keeps token indices aligned between both views.
    """
    lists: list[tuple[str, ...]] = []
    previous_ends_with_digit = False
    for original_token in original_tokens:
        folded = unicodedata.normalize("NFKC", original_token).casefold()
        if previous_ends_with_digit:
            with_context = normalize_text(f"0 {original_token}")
            context_tokens = tuple(with_context.split())
            if context_tokens and context_tokens[0] == "0":
                lists.append(context_tokens[1:])
            else:
                lists.append(tuple(normalize_text(original_token).split()))
        else:
            lists.append(tuple(normalize_text(original_token).split()))
        previous_ends_with_digit = _TRAILING_DIGIT_RE.search(folded) is not None
    return tuple(lists)


@lru_cache(maxsize=1024)
def _get_rehydration_candidate(candidate_text: str, language: str | None) -> Candidate:
    """Get or create a dummy Candidate instance cached by text and language."""
    return Candidate(text=candidate_text, intent_name="dummy", language=language)


def _find_rehydration_boundaries(
    candidate_tokens: tuple[str, ...],
    wc_idx: int,
    query_tokens: tuple[str, ...],
    wildcard_indices: frozenset[int] = frozenset(),
) -> tuple[int, int] | None:
    """Find query token indices for wildcard prefix and suffix boundaries."""
    q_len = len(query_tokens)

    if c_prefix := candidate_tokens[:wc_idx]:
        local_prefix_wc = frozenset(idx for idx in wildcard_indices if idx < wc_idx)
        prefix_boundary = _align_prefix_boundary(c_prefix, query_tokens, local_prefix_wc)
        if prefix_boundary == -1:
            c_suffix = candidate_tokens[wc_idx + 1 :]
            if len(c_prefix) > 1 or not c_suffix:
                return None
            max_prefix_search = q_len - len(c_suffix)
            prefix_boundary = min(len(c_prefix), max(0, max_prefix_search - 1))
    else:
        prefix_boundary = 0

    if c_suffix := candidate_tokens[wc_idx + 1 :]:
        local_suffix_wc = frozenset(idx - (wc_idx + 1) for idx in wildcard_indices if idx > wc_idx)
        suffix_boundary = _align_suffix_boundary(
            c_suffix, query_tokens, prefix_boundary, local_suffix_wc
        )
        if suffix_boundary == -1:
            return None
    else:
        suffix_boundary = q_len

    if prefix_boundary >= suffix_boundary:
        return None

    return prefix_boundary, suffix_boundary


def _extract_wc_value(
    query: str,
    query_tokens: tuple[str, ...],
    prefix_boundary: int,
    suffix_boundary: int,
) -> str | None:
    """Extract wildcard value substring from the original query."""
    wc_value = _extract_original_span(query, prefix_boundary, suffix_boundary)
    if not wc_value:
        if wc_value_tokens := query_tokens[prefix_boundary:suffix_boundary]:
            wc_value = " ".join(wc_value_tokens)
        else:
            return None
    return wc_value


def _surface_prefix_length(value: str, folded_prefix: str) -> int:
    """Return the surface char count whose casefold equals folded_prefix, or -1.

    Casefolding can change string length ("ß" folds to "ss"), so the
    surface slice boundary must be found by accumulating per-character
    fold lengths instead of reusing ``len(folded_prefix)``.
    """
    target = len(folded_prefix)
    total = 0
    for index in range(len(value)):
        total += len(value[index].casefold())
        if total >= target:
            if total == target and value[: index + 1].casefold() == folded_prefix:
                return index + 1
            return -1
    return -1


def _surface_suffix_length(value: str, folded_suffix: str) -> int:
    """Return the surface char count whose casefold equals folded_suffix, or -1."""
    target = len(folded_suffix)
    total = 0
    for count, index in enumerate(range(len(value) - 1, -1, -1), start=1):
        total += len(value[index].casefold())
        if total >= target:
            if total == target and value[index:].casefold() == folded_suffix:
                return count
            return -1
    return -1


def _trim_wildcard_overlaps(wc_value: str, c_tok: str, matched_wc: str) -> str:
    """Trim partial prefix and suffix overlaps of the wildcard placeholder token from wc_value."""
    wc_pos = c_tok.find(matched_wc)
    if wc_pos != -1:
        c_suffix_part = c_tok[wc_pos + len(matched_wc) :]
        if c_prefix_part := c_tok[:wc_pos]:
            prefix_length = _surface_prefix_length(wc_value, c_prefix_part.casefold())
            if prefix_length != -1:
                wc_value = wc_value[prefix_length:]
        if c_suffix_part:
            suffix_length = _surface_suffix_length(wc_value, c_suffix_part.casefold())
            if suffix_length != -1:
                wc_value = wc_value[:-suffix_length]
    return wc_value


def _has_adjacent_wildcards(wildcard_infos: tuple[tuple[int, str], ...]) -> bool:
    """Return whether any two wildcards occupy adjacent normalized token indices."""
    return any(
        next_index <= current_index + 1
        for (current_index, _), (next_index, _) in pairwise(wildcard_infos)
    )


def _extract_wildcard_replacement(
    norm_tokens: tuple[str, ...],
    wc_idx: int,
    wc_name: str,
    wildcard_indices: frozenset[int],
    query: str,
    query_tokens: tuple[str, ...],
) -> str | None:
    """Align, extract, and trim one wildcard value, or None when unresolvable.

    Stem alignment locates query token boundaries for this specific wildcard.
    Other registered wildcards are treated as position-agnostic wildcard tokens,
    which allows prefix/suffix stems to align successfully around overlapping
    wildcards. The raw matched substring is then extracted from the query, and
    overlapping literals are trimmed (e.g. if stem alignment fuzzily matched
    template suffix/prefix words).
    """
    boundaries = _find_rehydration_boundaries(
        norm_tokens,
        wc_idx,
        query_tokens,
        wildcard_indices,
    )
    if boundaries is None:
        return None
    prefix_boundary, suffix_boundary = boundaries

    wc_value = _extract_wc_value(query, query_tokens, prefix_boundary, suffix_boundary)
    if not wc_value:
        return None

    c_tok = norm_tokens[wc_idx]
    wc_value = _trim_wildcard_overlaps(wc_value, c_tok, wc_name)

    return wc_value if wc_value.strip() else None


def _apply_wildcard_replacements(
    candidate_text: str,
    replacements_list: list[tuple[int, str, str]],
    wildcard_count: int,
) -> tuple[str, dict[str, str]]:
    """Substitute extracted wildcard values into the original candidate text.

    Replacements are sorted descending by token index to substitute from right
    to left, which keeps indices of wildcards further left stable during
    replacements. Fails closed on partial rehydration: a leftover literal
    placeholder token ("timer_duration") in delegated text can match a live
    wildcard slot downstream and execute with the placeholder as the slot value.
    """
    replacements_list.sort(key=lambda x: x[0], reverse=True)

    current_text = candidate_text
    final_replacements: dict[str, str] = {}
    replaced_count = 0
    for wc_idx, wc_name, wc_value in replacements_list:
        new_text = _replace_wildcard_in_original(current_text, wc_idx, wc_value, wc_name)
        if new_text and new_text != current_text:
            current_text = new_text
            final_replacements[wc_name] = wc_value
            replaced_count += 1

    if replaced_count != wildcard_count:
        return candidate_text, {}

    return current_text, final_replacements


def get_wildcard_rehydration(
    candidate: Candidate,
    query: str,
    query_tokens: tuple[str, ...] | None = None,
) -> tuple[str, dict[str, str]]:
    """Compute rehydrated text and slot replacement mapping, supporting multiple wildcards.

    Uses a single-pass extraction based on stable original candidate token indices,
    replacing wildcards from right-to-left (descending index) to avoid loop-based
    re-tokenization and prevent desync. Adjacent wildcards fail closed because the
    query provides no deterministic boundary between their free-text values.
    """
    wildcard_infos = candidate.wildcard_infos
    if not wildcard_infos:
        return candidate.text, {}
    if _has_adjacent_wildcards(wildcard_infos):
        return candidate.text, {}

    if query_tokens is None:
        _, query_tokens = _normalize_and_split(query)

    wildcard_indices = frozenset(idx for idx, _ in wildcard_infos)
    norm_tokens = candidate.normalized_tokens

    replacements_list: list[tuple[int, str, str]] = []

    for wc_idx, wc_name in wildcard_infos:
        wc_value = _extract_wildcard_replacement(
            norm_tokens,
            wc_idx,
            wc_name,
            wildcard_indices,
            query,
            query_tokens,
        )
        if wc_value is None:
            continue
        replacements_list.append((wc_idx, wc_name, wc_value))

    if not replacements_list:
        return candidate.text, {}

    return _apply_wildcard_replacements(candidate.text, replacements_list, len(wildcard_infos))


def rehydrate_wildcard_text(candidate_text: str, query: str, language: str | None = None) -> str:
    """Replace wildcard placeholder tokens in *candidate_text* with spans from *query*.

    Algorithm: prefix/suffix stem alignment:

    1. Identify wildcard placeholder tokens in *candidate_text*.
    2. Split the candidate at wildcard positions into a prefix stem and a
       suffix stem (non-wildcard tokens before / after the wildcard).
    3. Align the prefix stem against the query's prefix tokens using
       positional character similarity to locate the boundary where the
       query's wildcard span begins.
    4. Align the suffix stem against the query's suffix tokens (in reverse)
       to locate where the wildcard span ends.
    5. Extract the query tokens between prefix boundary and suffix boundary
       as the wildcard value.
    6. Substitute the wildcard placeholder in *candidate_text* with the
       extracted value.

    If alignment fails at any boundary, the candidate is returned unchanged.
    Performance: O(T) per call where T = total tokens in candidate + query.
    """
    wildcards = wildcard_slot_names(language)
    if not wildcards or all(wc not in candidate_text for wc in wildcards):
        return candidate_text
    candidate = _get_rehydration_candidate(candidate_text, language)
    text, _ = get_wildcard_rehydration(candidate, query)
    return text


def rehydrate_wildcard_slots(
    slots: Mapping[str, JsonValueType],
    candidate_text: str,
    query: str,
    language: str | None = None,
) -> JsonObjectType:
    """Rehydrate wildcard values inside slots dictionary using query."""
    if not slots:
        return dict(slots)
    wildcards = wildcard_slot_names(language)
    if not wildcards or all(wc not in candidate_text for wc in wildcards):
        return dict(slots)
    candidate = _get_rehydration_candidate(candidate_text, language)
    _, replacements = get_wildcard_rehydration(candidate, query)
    if not replacements:
        return dict(slots)
    return {
        k: replacements[k] if isinstance(v, str) and k in replacements and v == k else v
        for k, v in slots.items()
    }


@lru_cache(maxsize=4096)
def _folded_with_offsets(o_tok: str) -> tuple[str, tuple[int, ...]]:
    """Return NFKC-casefolded text with original-index offsets per folded char.

    Normalized tokens come from ``normalize_text`` (NFKC + casefold), so
    matching against ``str.lower()`` breaks for casefold-divergent characters
    (``ß`` → ``ss``, ligatures, full-width forms). Folding per character keeps
    an index map back into the original string for slicing.
    """
    if o_tok.isascii():
        return o_tok.lower(), tuple(range(len(o_tok) + 1))
    parts: list[str] = []
    offsets: list[int] = []
    length = len(o_tok)
    index = 0
    while index < length:
        # Fold each base character together with its trailing combining
        # marks so decomposed (NFD) input composes to the NFC forms that
        # whole-string normalization produces; folding characters one at a
        # time would leave "e" + U+0301 decomposed and break `find`.
        end = index + 1
        while end < length and unicodedata.category(o_tok[end]).startswith("M"):
            end += 1
        folded = unicodedata.normalize("NFKC", o_tok[index:end]).casefold()
        parts.append(folded)
        offsets.extend([index] * len(folded))
        index = end
    offsets.append(length)
    return "".join(parts), tuple(offsets)


def _get_token_slice(
    o_tok: str, norm_tokens: tuple[str, ...] | list[str], start_norm_idx: int, end_norm_idx: int
) -> str:
    """Return slice of o_tok matching norm_tokens[start_norm_idx:end_norm_idx]."""
    if start_norm_idx == 0 and end_norm_idx == len(norm_tokens):
        return o_tok
    folded, offsets = _folded_with_offsets(o_tok)
    idx = 0
    start_char = 0
    for k in range(end_norm_idx):
        sub_idx = folded.find(norm_tokens[k], idx)
        if sub_idx == -1:
            break
        idx = sub_idx + len(norm_tokens[k])
        if k == start_norm_idx - 1:
            start_char = idx
    last_offset = len(offsets) - 1
    return o_tok[offsets[min(start_char, last_offset)] : offsets[min(idx, last_offset)]]


@lru_cache(maxsize=256)
def _query_norm_token_lists(
    original_query: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Return query tokens and their normalized token lists, cached per query.

    Ranking can rehydrate hundreds of wildcard candidates against the same
    query; this work depends only on the query text.
    """
    original_tokens = tuple(original_query.split())
    return original_tokens, _contextual_norm_token_lists(original_tokens)


@lru_cache(maxsize=1024)
def _candidate_norm_token_lists(
    candidate_text: str,
) -> tuple[tuple[str, ...], ...]:
    """Return contextual normalized token lists cached by candidate text."""
    return _contextual_norm_token_lists(tuple(candidate_text.split()))


def _extract_original_span(
    original_query: str,
    prefix_boundary: int,
    suffix_boundary: int,
) -> str:
    """Extract original case span from original_query for the normalized tokens slice."""
    original_tokens, norm_token_lists = _query_norm_token_lists(original_query)

    normalized_index = 0
    start_oi = -1
    start_sub_idx = -1
    end_oi = -1
    end_sub_idx = -1

    start_target = prefix_boundary
    end_target = suffix_boundary - 1

    for oi, o_tok_norm_tokens in enumerate(norm_token_lists):
        if not o_tok_norm_tokens:
            continue

        if normalized_index <= start_target < normalized_index + len(o_tok_norm_tokens):
            start_oi = oi
            start_sub_idx = start_target - normalized_index

        if normalized_index <= end_target < normalized_index + len(o_tok_norm_tokens):
            end_oi = oi
            end_sub_idx = end_target - normalized_index + 1

        normalized_index += len(o_tok_norm_tokens)

    if start_oi == -1 or end_oi == -1:
        return ""

    if start_oi == end_oi:
        o_tok = original_tokens[start_oi]
        o_tok_norm_tokens = norm_token_lists[start_oi]
        return _get_token_slice(o_tok, o_tok_norm_tokens, start_sub_idx, end_sub_idx)

    start_tok = original_tokens[start_oi]
    start_norm_tokens = norm_token_lists[start_oi]
    start_part = _get_token_slice(
        start_tok, start_norm_tokens, start_sub_idx, len(start_norm_tokens)
    )

    end_tok = original_tokens[end_oi]
    end_norm_tokens = norm_token_lists[end_oi]
    end_part = _get_token_slice(end_tok, end_norm_tokens, 0, end_sub_idx)

    middle_parts = list(original_tokens[start_oi + 1 : end_oi])

    parts = []
    if start_part:
        parts.append(start_part)
    parts.extend(middle_parts)
    if end_part:
        parts.append(end_part)
    return " ".join(parts)


def _token_similarity(c_tok: str, q_tok: str, is_wildcard: bool = False) -> float:
    """Compute similarity between a candidate token and a query token."""
    if c_tok == q_tok:
        return 1.0
    if is_wildcard:
        return 1.0
    len_c = len(c_tok)
    len_q = len(q_tok)
    min_len = min(len_c, len_q)
    max_len = max(len_c, len_q)
    if min_len == 0 or max_len >= MAX_TOKEN_LENGTH_RATIO * min_len:
        return 0.0
    return _cached_fuzz_ratio(c_tok, q_tok) / 100.0


def _best_alignment_start(
    candidate_tokens: tuple[str, ...],
    query_tokens: tuple[str, ...],
    starts: range,
    wildcard_indices: frozenset[int],
) -> int:
    """Return the strongest matching window start in the supplied priority order."""
    has_wildcard = not wildcard_indices.isdisjoint(range(len(candidate_tokens)))
    if not has_wildcard:
        for start in starts:
            if query_tokens[start : start + len(candidate_tokens)] == candidate_tokens:
                return start

    best_start = -1
    best_score = -1.0
    for start in starts:
        score = 0.0
        for index, candidate_token in enumerate(candidate_tokens):
            similarity = _token_similarity(
                candidate_token,
                query_tokens[start + index],
                is_wildcard=index in wildcard_indices,
            )
            if similarity < WILDCARD_STEM_ALIGNMENT_THRESHOLD:
                break
            score += similarity
        else:
            if score > best_score:
                best_start = start
                best_score = score
    return best_start


def _align_prefix_boundary(
    c_prefix: tuple[str, ...],
    query_tokens: tuple[str, ...],
    wildcard_indices: frozenset[int] = frozenset(),
) -> int:
    """Return the boundary after the strongest candidate-prefix alignment."""
    q_len = len(query_tokens)
    p_len = len(c_prefix)
    if p_len == 0:
        return 0
    max_start = q_len - p_len
    if max_start < 0:
        return -1
    start = _best_alignment_start(
        c_prefix,
        query_tokens,
        range(max_start + 1),
        wildcard_indices,
    )
    return -1 if start < 0 else min(start + p_len, q_len)


def _align_suffix_boundary(
    c_suffix: tuple[str, ...],
    query_tokens: tuple[str, ...],
    prefix_boundary: int,
    wildcard_indices: frozenset[int] = frozenset(),
) -> int:
    """Return the boundary before the strongest candidate-suffix alignment."""
    q_len = len(query_tokens)
    s_len = len(c_suffix)
    if s_len == 0:
        return q_len
    max_end = q_len - s_len
    if max_end < prefix_boundary:
        return -1
    return _best_alignment_start(
        c_suffix,
        query_tokens,
        range(max_end, prefix_boundary - 1, -1),
        wildcard_indices,
    )


@lru_cache(maxsize=256)
def _get_escaped_pattern(text: str) -> re.Pattern[str]:
    """Compile and return an escaped case-insensitive regex pattern."""
    return re.compile(re.escape(text), re.IGNORECASE)


def _replace_wildcard_in_original(
    original_text: str,
    wildcard_normalized_index: int,
    replacement: str,
    matched_wc: str | None = None,
) -> str:
    """Replace the wildcard token in *original_text* with *replacement*.

    Maps the wildcard's position in the normalized token sequence back to
    the original (non-casefolded, non-NFKC) text and performs a literal
    substring substitution.
    """
    original_tokens = original_text.split()
    norm_token_lists = _candidate_norm_token_lists(original_text)
    normalized_index = 0
    for oi, o_tok in enumerate(original_tokens):
        o_tok_norm_tokens = norm_token_lists[oi]
        if not o_tok_norm_tokens:
            continue
        if (
            normalized_index
            <= wildcard_normalized_index
            < normalized_index + len(o_tok_norm_tokens)
        ):
            sub_idx = wildcard_normalized_index - normalized_index
            target_placeholder = matched_wc or o_tok_norm_tokens[sub_idx]
            if match := _get_escaped_pattern(target_placeholder).search(o_tok):
                start, end = match.span()
                original_tokens[oi] = o_tok[:start] + replacement + o_tok[end:]
                return " ".join(original_tokens)
        normalized_index += len(o_tok_norm_tokens)

    return original_text


def wildcard_variants_analysis(
    candidate: Candidate,
) -> tuple[tuple[WildcardVariantAnalysis, ...], frozenset[str]]:
    """Compute wildcard literal-coverage requirements and return all literal tokens."""
    variants = candidate.literal_variants
    wc_names = tuple(name for _, name in candidate.wildcard_infos)
    return _wildcard_variants_analysis_cached(variants, wc_names)


@lru_cache(maxsize=2048)
def _wildcard_variants_analysis_cached(
    variants: tuple[frozenset[str], ...],
    wc_names: tuple[str, ...],
) -> tuple[tuple[WildcardVariantAnalysis, ...], frozenset[str]]:
    """Return wildcard literal coverage data for a literal variant/wildcard pair."""
    variant_analyses: list[WildcardVariantAnalysis] = []
    seen_variants: set[tuple[frozenset[str], int]] = set()
    all_tokens_set = set()
    all_variant_tokens = {tok for var in variants for tok in var} if wc_names else set()
    variant_set = frozenset(variants) if wc_names else frozenset()
    for var in variants:
        clean_var = (
            frozenset(
                tok
                for tok in var
                if not any(
                    _is_wildcard_literal_token(
                        tok,
                        wc_name,
                        all_variant_tokens,
                        var,
                        variant_set,
                    )
                    for wc_name in wc_names
                )
            )
            if wc_names
            else var
        )
        length = len(clean_var)
        req = ceil(length * WILDCARD_LITERAL_COVERAGE_THRESHOLD)
        variant_key = (clean_var, req)
        if variant_key in seen_variants:
            continue
        seen_variants.add(variant_key)
        variant_analyses.append(
            WildcardVariantAnalysis(
                literal_tokens=clean_var,
                required_match_count=req,
            )
        )
        all_tokens_set.update(clean_var)
    return tuple(variant_analyses), frozenset(all_tokens_set)


def _is_wildcard_literal_token(
    token: str,
    wc_name: str,
    all_variant_tokens: set[str],
    variant: frozenset[str],
    variants: frozenset[frozenset[str]],
) -> bool:
    """Return whether a literal token still carries a wildcard placeholder."""
    if token == wc_name:
        return True
    if wc_name not in token:
        return False
    if "_" in wc_name:
        return True

    idx = token.find(wc_name)
    prefix = token[:idx]
    suffix = token[idx + len(wc_name) :]

    if (prefix in all_variant_tokens) or (suffix in all_variant_tokens):
        return True
    variant_without_token = variant - {token}
    return bool(variant_without_token and variant_without_token in variants)


def clear_rehydration_caches() -> None:
    """Clear all global LRU caches in rehydration module."""
    _get_rehydration_candidate.cache_clear()
    _get_escaped_pattern.cache_clear()
    _wildcard_variants_analysis_cached.cache_clear()
    _raw_cached_fuzz_ratio.cache_clear()
    _folded_with_offsets.cache_clear()
    _query_norm_token_lists.cache_clear()
    _candidate_norm_token_lists.cache_clear()
