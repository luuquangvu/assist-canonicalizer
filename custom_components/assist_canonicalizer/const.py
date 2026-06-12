"""Constants for Assist Canonicalizer."""

from enum import StrEnum

DOMAIN = "assist_canonicalizer"
NAME = "Assist Canonicalizer"

CONF_FALLBACK_AGENT_ID = "fallback_agent_id"
CONF_MIN_CONFIDENCE = "min_confidence"
CONF_MIN_MARGIN = "min_margin"

DATA_RUNTIME = "runtime"

SERVICE_CLEAR_INDEX = "clear_index"
SERVICE_DIAGNOSTICS = "diagnostics"
SERVICE_DUMP_CANDIDATES = "dump_candidates"
SERVICE_REBUILD_INDEX = "rebuild_index"
SERVICE_TEST_MATCH = "test_match"

ATTR_ACCEPTED = "accepted"
ATTR_CANDIDATE_COUNT = "candidate_count"
ATTR_INTENT_NAME = "intent_name"
ATTR_LANGUAGE = "language"
ATTR_NORMALIZED_TEXT = "normalized_text"
ATTR_SELECTED_CANDIDATE = "selected_candidate"
ATTR_SOURCE = "source"
ATTR_TEXT = "text"
ATTR_TOP_CANDIDATES = "top_candidates"

DEFAULT_MIN_CONFIDENCE = 0.50
DEFAULT_MIN_MARGIN = 0.04
DEFAULT_MAX_CANDIDATES = 20
DEFAULT_VALIDATION_CANDIDATES = 5

DEFAULT_MAX_TOTAL_CANDIDATES_PER_LANGUAGE = 60000
DEFAULT_MAX_CANDIDATES_PER_INTENT = 1000
DEFAULT_MAX_CANDIDATES_PER_TEMPLATE = 200
DEFAULT_MAX_DYNAMIC_SLOT_VALUES = 20
DEFAULT_MAX_DYNAMIC_CANDIDATES = 200
DEFAULT_RAPIDFUZZ_PREFILTER_CANDIDATES = 500

RAPIDFUZZ_WEIGHT = 0.33
CHAR_NGRAM_WEIGHT = 0.32
BM25_WEIGHT = 0.20
INTENT_ACTION_WEIGHT: float = 1 - (RAPIDFUZZ_WEIGHT + CHAR_NGRAM_WEIGHT + BM25_WEIGHT)

POSITIONAL_SIMILARITY_THRESHOLD = 0.65
POSITIONAL_SIMILARITY_PARTIAL_CREDIT = 0.50


class FallbackReason(StrEnum):
    """Conversation fallback reason values for diagnostics."""

    LOW_CONFIDENCE = "low_confidence"
    LOW_MARGIN = "low_margin"
    EMPTY_INDEX = "empty_index"
    VALIDATION_FAILED = "validation_failed"
    RANKING_FAILED = "ranking_failed"
    UNEXPECTED_EXCEPTION = "unexpected_exception"


# GENERIC_LATIN_REPLACEMENTS maps specific Latin-extended characters
# to ASCII/simpler equivalents.
GENERIC_LATIN_REPLACEMENTS = {
    "đ": "d",
    "ß": "ss",
    "æ": "ae",
    "œ": "oe",
    "ø": "o",
    "ł": "l",
    "ı": "i",  # noqa: RUF001
    "ð": "d",
    "þ": "th",
}

# LANGUAGE_SPECIFIC_OVERRIDES maps ISO language codes (e.g., "de") to
# character replacement mappings. These overrides take priority over
# global/default mappings when canonicalizing text for the given language.
LANGUAGE_SPECIFIC_OVERRIDES = {
    "de": {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
    }
}
