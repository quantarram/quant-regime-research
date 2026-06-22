"""
economic_prior_BYPASS.py
=========================
A drop-in replacement for economic_prior.py that admits every (X, Y)
pair unconditionally. Same function signatures, so cpe_engine_parallel.py
and joint_cpe_engine.py can import this instead with zero code changes,
purely to produce the UNRESTRICTED comparison arm of the prior-vs-no-prior
A/B test.

This is NOT a "better" or "simpler" version of the prior -- it is
deliberately the null version, used only to regenerate the same screen
your original (pre-prior) pipeline would have produced, so it can be run
through the identical backtest code as the prior-gated version for a
fair, same-code comparison.

Usage: see RUN_ORDER.md. In short, set:
    export USE_PRIOR_BYPASS=1
before running cpe_engine_parallel.py and joint_cpe_engine.py, and they
will import this module instead of economic_prior.py.
"""

from typing import Dict, Tuple


def get_subclass(ticker: str) -> str:
    return "unrestricted"


def is_admissible(predictor: str, target: str, min_confidence: str = "weak") -> bool:
    """Admits everything except a ticker predicting itself. This is the
    null prior -- no economic restriction, no duplicate-instrument
    exclusion, no confidence tiering. Used only to regenerate the
    unrestricted comparison arm."""
    return predictor != target


def explain_pair(predictor: str, target: str) -> str:
    if predictor == target:
        return "REJECTED: predictor and target are the same instrument."
    return "ADMISSIBLE [bypass mode]: no economic prior applied."


def admissible_predictors_for(target: str, universe: list, min_confidence: str = "weak") -> list:
    return [x for x in universe if x != target]


def get_confidence(predictor_subclass: str, target_subclass: str) -> str:
    return "standard"


# Empty stand-ins so any code that imports these names directly doesn't break
ADMISSIBLE_CHANNELS: Dict[Tuple[str, str], str] = {}
CONFIDENCE_OVERRIDES: Dict[Tuple[str, str], str] = {}
SUBCLASS: Dict[str, str] = {}
