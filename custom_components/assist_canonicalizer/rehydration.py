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


def _normalize_and_split(text: str) -> tuple[str, tuple[str, ...]]:
    """Normalize text and return both the normalized text and its tokens tuple."""
    norm = normalize_text(text)
    return norm, tuple(norm.split())


@lru_cache(maxsize=1024)
def _get_rehydration_candidate(candidate_text: str, language: str | None) -> Candidate:
    """Get or create a dummy Candidate instance cached by text and language."""
    return Candidate(text=candidate_text, intent_name="dummy", language=language)


def get_wildcard_rehydration(
    candidate: Candidate,
    query: str,
    query_tokens: tuple[str, ...] | None = None,
) -> tuple[str, dict[str, str]]:
    """Compute rehydrated text and slot replacement mapping."""
    wildcard_info = candidate.wildcard_info
    if wildcard_info is None:
        return candidate.text, {}

    wc_idx, matched_wc = wildcard_info
    if query_tokens is None:
        _, query_tokens = _normalize_and_split(query)

    candidate_tokens = candidate.normalized_tokens
    c_prefix = candidate_tokens[:wc_idx]
    q_len = len(query_tokens)

    if c_prefix:
        prefix_boundary = _align_prefix_boundary(c_prefix, query_tokens)
        if prefix_boundary == -1:
            if len(c_prefix) <= 1:
                c_suffix = candidate_tokens[wc_idx + 1 :]
                max_prefix_search = q_len - len(c_suffix)
                prefix_boundary = min(len(c_prefix), max(0, max_prefix_search - 1))
            else:
                return candidate.text, {}
    else:
        prefix_boundary = 0

    c_suffix = candidate_tokens[wc_idx + 1 :]

    if c_suffix:
        suffix_boundary = _align_suffix_boundary(c_suffix, query_tokens, prefix_boundary)
        if suffix_boundary == -1:
            return candidate.text, {}
    else:
        suffix_boundary = q_len

    if prefix_boundary >= suffix_boundary:
        return candidate.text, {}

    wc_value = _extract_original_span(query, prefix_boundary, suffix_boundary)
    if not wc_value:
        wc_value_tokens = query_tokens[prefix_boundary:suffix_boundary]
        if not wc_value_tokens:
            return candidate.text, {}
        wc_value = " ".join(wc_value_tokens)

    c_tok = candidate_tokens[wc_idx]
    wc_pos = c_tok.find(matched_wc)
    if wc_pos != -1:
        c_prefix_part = c_tok[:wc_pos]
        c_suffix_part = c_tok[wc_pos + len(matched_wc) :]
        if c_prefix_part and wc_value.lower().startswith(c_prefix_part.lower()):
            wc_value = wc_value[len(c_prefix_part) :]
        if c_suffix_part and wc_value.lower().endswith(c_suffix_part.lower()):
            wc_value = wc_value[: -len(c_suffix_part)]

    if not wc_value.strip():
        return candidate.text, {}

    result = _replace_wildcard_in_original(candidate.text, wc_idx, wc_value, matched_wc)
    if not result or result == candidate.text:
        return candidate.text, {}

    return result, {matched_wc: wc_value}


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
    if not wildcards or not any(wc in candidate_text for wc in wildcards):
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
    if not wildcards or not any(wc in candidate_text for wc in wildcards):
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


def _token_similarity(c_tok: str, q_tok: str) -> float:
    """Compute similarity between a candidate token and a query token."""
    if c_tok == q_tok:
        return 1.0
    len_c = len(c_tok)
    len_q = len(q_tok)
    min_len = len_c if len_c < len_q else len_q
    max_len = len_c if len_c > len_q else len_q
    if min_len == 0 or max_len >= MAX_TOKEN_LENGTH_RATIO * min_len:
        return 0.0
    return float(fuzz.ratio(c_tok, q_tok)) / 100.0


def _align_prefix_boundary(
    c_prefix: tuple[str, ...],
    query_tokens: tuple[str, ...],
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
    if q_len >= p_len and query_tokens[:p_len] == c_prefix:
        return p_len
    max_start = q_len - p_len
    if max_start < 0:
        return -1

    # Fast path: exact sub-slice match of c_prefix in query_tokens
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

        sim = _token_similarity(c_tok, q_tok)

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

    return best_boundary if best_boundary <= q_len else q_len


def _align_suffix_boundary(
    c_suffix: tuple[str, ...],
    query_tokens: tuple[str, ...],
    prefix_boundary: int,
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
    if q_len - prefix_boundary >= s_len and query_tokens[-s_len:] == c_suffix:
        return q_len - s_len
    max_end = q_len - s_len
    if max_end < prefix_boundary:
        return -1

    # Fast path: exact sub-slice match of c_suffix in query_tokens (rightmost first)
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

        sim = _token_similarity(c_tok, q_tok)

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
            target_placeholder = matched_wc if matched_wc else o_tok_norm_tokens[sub_idx]
            match = re.search(re.escape(target_placeholder), o_tok, re.IGNORECASE)
            if match:
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
    wc_info = candidate.wildcard_info
    wc_name = wc_info[1] if wc_info else None

    var_with_len = []
    all_tokens_set = set()
    for var in variants:
        clean_var = frozenset(tok for tok in var if tok != wc_name) if wc_name else var
        length = len(clean_var)
        req = ceil(length * WILDCARD_LITERAL_COVERAGE_THRESHOLD)
        var_with_len.append((clean_var, length, req))
        all_tokens_set.update(clean_var)
    return tuple(var_with_len), frozenset(all_tokens_set)
