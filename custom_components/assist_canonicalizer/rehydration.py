"""Dynamic wildcard rehydration and stem alignment logic."""

from __future__ import annotations

import re
from functools import lru_cache
from math import ceil
from typing import Any

from rapidfuzz import fuzz

from .candidate import Candidate
from .const import (
    MAX_TOKEN_LENGTH_RATIO,
    WILDCARD_LITERAL_COVERAGE_THRESHOLD,
    WILDCARD_STEM_ALIGNMENT_THRESHOLD,
)
from .normalization import normalize_text
from .utils import wildcard_slot_names


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


def _trim_wildcard_overlaps(wc_value: str, c_tok: str, matched_wc: str) -> str:
    """Trim partial prefix and suffix overlaps of the wildcard placeholder token from wc_value."""
    wc_pos = c_tok.find(matched_wc)
    if wc_pos != -1:
        c_prefix_part = c_tok[:wc_pos]
        c_suffix_part = c_tok[wc_pos + len(matched_wc) :]
        if c_prefix_part and wc_value.lower().startswith(c_prefix_part.lower()):
            wc_value = wc_value[len(c_prefix_part) :]
        if c_suffix_part and wc_value.lower().endswith(c_suffix_part.lower()):
            wc_value = wc_value[: -len(c_suffix_part)]
    return wc_value


def get_wildcard_rehydration(
    candidate: Candidate,
    query: str,
    query_tokens: tuple[str, ...] | None = None,
) -> tuple[str, dict[str, str]]:
    """Compute rehydrated text and slot replacement mapping, supporting multiple wildcards.

    Uses a single-pass extraction based on stable original candidate token indices,
    replacing wildcards from right-to-left (descending index) to avoid loop-based
    re-tokenization and prevent desync.

    Known limitation: when two wildcards are adjacent (no literal separator tokens
    between them), each wildcard's boundary is computed independently and may claim
    overlapping query token spans.  In practice, templates include separator tokens
    between wildcards (e.g. "play {song} by {artist}"), so this does not surface.
    """
    wildcard_infos = candidate.wildcard_infos
    if not wildcard_infos:
        return candidate.text, {}

    if query_tokens is None:
        _, query_tokens = _normalize_and_split(query)

    wildcard_indices = frozenset(idx for idx, _ in wildcard_infos)
    norm_tokens = candidate.normalized_tokens

    replacements_list: list[tuple[int, str, str]] = []

    for wc_idx, wc_name in wildcard_infos:
        # 1. Align stems to find query token boundaries for this specific wildcard.
        # Other registered wildcards are treated as position-agnostic wildcard tokens,
        # which allows prefix/suffix stems to align successfully around overlapping wildcards.
        boundaries = _find_rehydration_boundaries(
            norm_tokens,
            wc_idx,
            query_tokens,
            wildcard_indices,
        )
        if boundaries is None:
            continue
        prefix_boundary, suffix_boundary = boundaries

        # 2. Extract the raw matched substring from the query.
        wc_value = _extract_wc_value(query, query_tokens, prefix_boundary, suffix_boundary)
        if not wc_value:
            continue

        # 3. Trim overlapping literals (e.g. if stem alignment fuzzily
        # matched template suffix/prefix words).
        c_tok = norm_tokens[wc_idx]
        wc_value = _trim_wildcard_overlaps(wc_value, c_tok, wc_name)

        if not wc_value.strip():
            continue

        replacements_list.append((wc_idx, wc_name, wc_value))

    if not replacements_list:
        return candidate.text, {}

    # Sort replacements descending by token index to substitute from right to left.
    # This ensures indices of wildcards further left remain stable during replacements.
    replacements_list.sort(key=lambda x: x[0], reverse=True)

    current_text = candidate.text
    final_replacements = {}
    for wc_idx, wc_name, wc_value in replacements_list:
        new_text = _replace_wildcard_in_original(current_text, wc_idx, wc_value, wc_name)
        if new_text and new_text != current_text:
            current_text = new_text
            final_replacements[wc_name] = wc_value

    return current_text, final_replacements


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
    slots: dict[str, Any], candidate_text: str, query: str, language: str | None = None
) -> dict[str, Any]:
    """Rehydrate wildcard values inside slots dictionary using query."""
    if not slots:
        return slots
    wildcards = wildcard_slot_names(language)
    if not wildcards or all(wc not in candidate_text for wc in wildcards):
        return slots
    candidate = _get_rehydration_candidate(candidate_text, language)
    _, replacements = get_wildcard_rehydration(candidate, query)
    if not replacements:
        return slots
    return {
        k: replacements[k] if isinstance(v, str) and k in replacements and v == k else v
        for k, v in slots.items()
    }


def _get_token_slice(
    o_tok: str, norm_tokens: list[str], start_norm_idx: int, end_norm_idx: int
) -> str:
    """Return slice of o_tok matching norm_tokens[start_norm_idx:end_norm_idx]."""
    if start_norm_idx == 0 and end_norm_idx == len(norm_tokens):
        return o_tok
    idx = 0
    start_char = 0
    o_tok_lower = o_tok.lower()
    for k in range(end_norm_idx):
        sub_idx = o_tok_lower.find(norm_tokens[k], idx)
        if sub_idx == -1:
            break
        idx = sub_idx + len(norm_tokens[k])
        if k == start_norm_idx - 1:
            start_char = idx
    return o_tok[start_char:idx]


def _extract_original_span(
    original_query: str,
    prefix_boundary: int,
    suffix_boundary: int,
) -> str:
    """Extract original case span from original_query for the normalized tokens slice."""
    original_tokens = original_query.split()

    norm_token_lists = [normalize_text(tok).split() for tok in original_tokens]

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

    middle_parts = original_tokens[start_oi + 1 : end_oi]

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


def _align_prefix_boundary(
    c_prefix: tuple[str, ...],
    query_tokens: tuple[str, ...],
    wildcard_indices: frozenset[int] = frozenset(),
) -> int:
    """Return the query token index where the wildcard span begins.

    Aligns the candidate prefix tokens against a sliding window of the query
    starting at position 0.  Each candidate token must positionally match a
    query token at the corresponding offset.  Returns the index immediately
    after the matched window, or -1 if alignment fails.
    """
    q_len = len(query_tokens)
    p_len = len(c_prefix)
    if p_len == 0:
        return 0
    has_wildcard = any(i in wildcard_indices for i in range(p_len))
    if not has_wildcard and q_len >= p_len and query_tokens[:p_len] == c_prefix:
        return p_len
    max_start = q_len - p_len
    if max_start < 0:
        return -1

    # Fast path: exact sub-slice match of c_prefix in query_tokens
    if not has_wildcard:
        for start_idx in range(max_start + 1):
            if query_tokens[start_idx : start_idx + p_len] == c_prefix:
                return start_idx + p_len

    best_boundary = -1
    best_score = -1.0

    start = 0
    i = 0
    score = 0.0
    while start <= max_start:
        q_tok = query_tokens[start + i]
        c_tok = c_prefix[i]

        is_wc = i in wildcard_indices
        sim = _token_similarity(c_tok, q_tok, is_wildcard=is_wc)

        if sim < WILDCARD_STEM_ALIGNMENT_THRESHOLD:
            start += 1
            i = 0
            score = 0.0
            continue

        score += sim
        i += 1

        if i == p_len:
            if score > best_score:
                best_score = score
                best_boundary = start + p_len
            start += 1
            i = 0
            score = 0.0

    return min(best_boundary, q_len)


def _align_suffix_boundary(
    c_suffix: tuple[str, ...],
    query_tokens: tuple[str, ...],
    prefix_boundary: int,
    wildcard_indices: frozenset[int] = frozenset(),
) -> int:
    """Return the query token index where the wildcard span ends.

    Aligns the candidate suffix tokens in reverse against the query tokens
    starting from the end.  Returns the index (from the left) immediately
    before the matched window, or -1 if alignment fails.
    """
    q_len = len(query_tokens)
    s_len = len(c_suffix)
    if s_len == 0:
        return q_len
    has_wildcard = any(i in wildcard_indices for i in range(s_len))
    if not has_wildcard and q_len - prefix_boundary >= s_len and query_tokens[-s_len:] == c_suffix:
        return q_len - s_len
    max_end = q_len - s_len
    if max_end < prefix_boundary:
        return -1

    # Fast path: exact sub-slice match of c_suffix in query_tokens (rightmost first)
    if not has_wildcard:
        for end_idx in range(max_end, prefix_boundary - 1, -1):
            if query_tokens[end_idx : end_idx + s_len] == c_suffix:
                return end_idx

    best_boundary = -1
    best_score = -1.0

    end = max_end
    i = 0
    score = 0.0
    while end >= prefix_boundary:
        q_tok = query_tokens[end + i]
        c_tok = c_suffix[i]

        is_wc = i in wildcard_indices
        sim = _token_similarity(c_tok, q_tok, is_wildcard=is_wc)

        if sim < WILDCARD_STEM_ALIGNMENT_THRESHOLD:
            end -= 1
            i = 0
            score = 0.0
            continue

        score += sim
        i += 1

        if i == s_len:
            if score > best_score:
                best_score = score
                best_boundary = end
            end -= 1
            i = 0
            score = 0.0

    return best_boundary if best_boundary >= prefix_boundary else -1


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
    normalized_index = 0
    for oi, o_tok in enumerate(original_tokens):
        o_tok_norm = normalize_text(o_tok)
        o_tok_norm_tokens = o_tok_norm.split()
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
) -> tuple[tuple[tuple[frozenset[str], int, int], ...], frozenset[str]]:
    """Compute wildcard variants with length/requirement checks and return a set of all tokens."""
    variants = candidate.literal_variants
    wc_names = tuple(name for _, name in candidate.wildcard_infos)
    return _wildcard_variants_analysis_cached(variants, wc_names)


@lru_cache(maxsize=2048)
def _wildcard_variants_analysis_cached(
    variants: tuple[frozenset[str], ...],
    wc_names: tuple[str, ...],
) -> tuple[tuple[tuple[frozenset[str], int, int], ...], frozenset[str]]:
    """Return wildcard literal coverage data for a literal variant/wildcard pair."""
    var_with_len = []
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
        var_with_len.append((clean_var, length, req))
        all_tokens_set.update(clean_var)
    return tuple(var_with_len), frozenset(all_tokens_set)


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
