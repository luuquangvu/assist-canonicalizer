"""Constants for Assist Canonicalizer."""

from enum import StrEnum
from math import ceil

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
SERVICE_SET_FALLBACK_AGENT = "set_fallback_agent"
SERVICE_TEST_MATCH = "test_match"

ATTR_ACCEPTED = "accepted"
ATTR_AGENT_ID = "agent_id"
ATTR_CANDIDATE_COUNT = "candidate_count"
ATTR_INTENT_NAME = "intent_name"
ATTR_LANGUAGE = "language"
ATTR_NORMALIZED_TEXT = "normalized_text"
ATTR_SELECTED_CANDIDATE = "selected_candidate"
ATTR_SOURCE = "source"
ATTR_TEXT = "text"
ATTR_TOP_CANDIDATES = "top_candidates"

ASSISTANT_CONVERSATION = "conversation"
ENTITY_SLOT_NAMES = ("name", "entity", "entity_name")
AREA_SLOT_NAMES = ("area", "area_name")
FLOOR_SLOT_NAMES = ("floor", "floor_name")
LOCATION_SLOT_NAMES = (*AREA_SLOT_NAMES, *FLOOR_SLOT_NAMES)

ENTITY_SLOT_NAME_SET = frozenset(ENTITY_SLOT_NAMES)
LOCATION_SLOT_NAME_SET = frozenset(LOCATION_SLOT_NAMES)

PRESERVED_UNIT_SUFFIXES = {
    "°": "T_DEGREE",
    "%": "P_PERCENT",
}

DEFAULT_MIN_CONFIDENCE = 0.50
DEFAULT_MIN_MARGIN = 0.04
HIGH_CONFIDENCE_RELAXED_MIN_SCORE = 0.80
HIGH_CONFIDENCE_RELAXED_MIN_MARGIN = 0.005
SAFE_EMPTY_SLOT_RELAXED_MIN_MARGIN = 0.035
SAFE_EMPTY_SLOT_RELAXED_MIN_SCORE = 0.60
SAFE_INTENT_EVIDENCE_MAX_SCORE = 0.74
SAFE_INTENT_EVIDENCE_MIN_SCORE = 0.57
SAFE_INTENT_EVIDENCE_MIN_ADVANTAGE = 0.09
ZERO_INTENT_EVIDENCE_MAX_SCORE = 0.70
ZERO_INTENT_EVIDENCE_MAX_MARGIN = 0.07
DEFAULT_MAX_CANDIDATES = 20
DEFAULT_MAX_PREFLIGHT_ATTEMPTS = 3

DEFAULT_MAX_TOTAL_CANDIDATES_PER_LANGUAGE = 150000
DEFAULT_MAX_CANDIDATES_PER_INTENT = 5000
DEFAULT_MAX_CANDIDATES_PER_TEMPLATE = 150
DEFAULT_MAX_DYNAMIC_SLOT_VALUES = 20
DEFAULT_MAX_DYNAMIC_CANDIDATES = 200
DEFAULT_MAX_REGISTRY_VALUES_NOMINATED = 64
DEFAULT_MAX_REGISTRY_VALUES_SCORED_PER_QUERY = 128
REGISTRY_FUZZY_TOKEN_MIN_LENGTH = 5
DEFAULT_RAPIDFUZZ_PREFILTER_CANDIDATES = 500
DEFAULT_SLOT_PREFILTER_RESCUE_CANDIDATES = 32
# Wildcard rescue candidates run the most expensive per-candidate scoring path
# (stem-alignment rehydration plus rescoring), so their budget is much smaller
# than the general prefilter. Selection is by literal-coverage relevance.
DEFAULT_WILDCARD_PREFILTER_RESCUE_CANDIDATES = 64

RAPIDFUZZ_WEIGHT = 0.33
CHAR_NGRAM_WEIGHT = 0.32
BM25_WEIGHT = 0.20
INTENT_ACTION_WEIGHT: float = 1 - (RAPIDFUZZ_WEIGHT + CHAR_NGRAM_WEIGHT + BM25_WEIGHT)

# Adaptive positional similarity thresholds by min(len(a),len(b)).
# Shorter tokens are more susceptible to false-positive positional matches, so
# they need stricter (higher) thresholds.
POSITIONAL_SIMILARITY_VERY_SHORT_THRESHOLD = 0.75  # per-pair min_len <= 2
POSITIONAL_SIMILARITY_SHORT_3_THRESHOLD = 0.66  # per-pair min_len <= 3
POSITIONAL_SIMILARITY_MEDIUM_THRESHOLD = 0.60  # per-pair min_len <= 5
POSITIONAL_SIMILARITY_BASE_THRESHOLD = 0.50  # per-pair min_len >= 6
POSITIONAL_FUZZY_TOKEN_MAX_LENGTH_RATIO = 1.6
SLOT_FUZZY_TOKEN_MIN_LENGTH = 4

# Partial credit awarded when a literal token positionally matches a query token.
POSITIONAL_SIMILARITY_PARTIAL_CREDIT = 0.50

# Non-entity penalty blend: fraction of the intent score that can be reduced
# when a candidate fails to cover non-entity query tokens (politeness/filler).
NON_ENTITY_PENALTY_BLEND = 0.05

# Minimum fraction of literal variant tokens that must match query tokens
# to evaluate a wildcard candidate.
WILDCARD_LITERAL_COVERAGE_THRESHOLD = 0.75

# Minimum character-level similarity required to align a template prefix/suffix
# token with a query token during wildcard rehydration.
WILDCARD_STEM_ALIGNMENT_THRESHOLD = 0.60

# Maximum acceptable length ratio between tokens to proceed with similarity computation.
# Derived from WILDCARD_STEM_ALIGNMENT_THRESHOLD: if max_len / min_len <= ratio,
# then fuzz.ratio >= threshold must hold. We use ceil to provide a safety margin
# against floating-point rounding errors.
MAX_TOKEN_LENGTH_RATIO = ceil(((2.0 / WILDCARD_STEM_ALIGNMENT_THRESHOLD) - 1.0) * 10000) / 10000

# Penalty factor per word in rehydrated wildcard to favor templates with more literal tokens
WILDCARD_LENGTH_PENALTY_FACTOR = 0.015

# Slot-token fuzzy matching uses RapidFuzz's 0-100 ratio. The 75 threshold is
# loose enough for minor inflection, pluralization, and diacritic differences,
# but strict enough that short location/entity words still need most characters
# to agree. Tune this with the structured penalties below because the same
# match gate controls unanchored-entity and static-slot conflict detection.
SLOT_TOKEN_MATCH_THRESHOLD = 75

# Structured ranking adjustments. These are intentionally language-neutral:
# they use candidate slot metadata, intent context, and numeric-slot shape
# rather than localized phrases. Boosts are additive and penalties are
# multiplicative, so lowering a penalty has a larger effect on already weak
# lexical matches than on high-confidence matches.
CONTEXT_SLOT_MATCH_BOOST = 0.10
CONTEXT_SLOT_MISMATCH_PENALTY = 0.85
ENTITY_ONLY_UNCOVERED_QUERY_PENALTY = 0.72
NUMERIC_SLOT_WITHOUT_QUERY_PENALTY = 0.78
NUMERIC_LITERAL_WITHOUT_QUERY_PENALTY = 0.64
NUMERIC_SLOT_MISMATCH_PENALTY = 0.05
QUERY_NUMBER_WITHOUT_CANDIDATE_PENALTY = 0.78
STATIC_ENTITY_UNCOVERED_QUERY_PENALTY = 0.82
STATIC_SLOT_QUERY_CONFLICT_PENALTY = 0.74
UNANCHORED_ENTITY_SLOT_PENALTY = 0.88
UNANCHORED_SEMANTIC_SLOT_PENALTY = 0.78
WILDCARD_KNOWN_SLOT_TOKEN_PENALTY = 0.78


class FallbackReason(StrEnum):
    """Conversation fallback reason values for diagnostics."""

    LOW_CONFIDENCE = "low_confidence"
    LOW_MARGIN = "low_margin"
    EMPTY_INDEX = "empty_index"
    VALIDATION_FAILED = "validation_failed"
    RANKING_FAILED = "ranking_failed"
    UNEXPECTED_EXCEPTION = "unexpected_exception"
    NO_CANDIDATE = "no_candidate"


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
