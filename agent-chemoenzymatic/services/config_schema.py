"""Config schema validation for built-in config and override files.

Pydantic-backed. The public API (``validate_config(raw, strict=True)``
and ``ConfigValidationError``) is preserved from the previous frozenset
allow-list implementation used before the runtime moved under ``core``.

Goal (unchanged from the previous implementation): catch typos in YAML
keys early so a stray underscore (the original bug: `max_iterations_:`
instead of `max_iterations:`) surfaces as a startup error instead of
silently falling back to the hardcoded default and producing eval
results nobody can explain.

Strategy: validate the raw dict against ``services.config_models.AgentConfig``
(extra="forbid" at the root and inside every well-modelled block).
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from services.config_models import AgentConfig

logger = logging.getLogger(__name__)

# Top-level block keys whose YAML value may legitimately be `null` (an
# author commenting the whole block out without deleting the heading).
# A null block is not a typo — it's "use defaults" — so we normalise it
# to `{}` before handing to the pydantic schema, which declares each of
# these as a non-Optional model with a default factory.
NORMALISABLE_BLOCK_KEYS = frozenset({
    "llm", "agent", "scratchpad", "prompts", "result_store",
    "postprocess", "templates", "suggest_templates", "stocks", "neural_retro",
    "enzymatic_retro", "evaluation", "verifier", "compaction",
    "precedent_recall",
})


def normalise_null_blocks(raw: dict) -> dict:
    """Return a shallow copy of ``raw`` where any known block whose value
    is ``None`` (a commented-out heading like ``agent:``) is replaced with
    ``{}``.

    Both ``validate_config`` here and ``run.load_config`` (the projector)
    rely on these blocks being dict-shaped — without this coercion the
    projector's ``raw.get("agent", {}).get(...)`` chain crashes with
    ``AttributeError: 'NoneType' object has no attribute 'get'`` at
    startup even though validation would have passed. Sharing one helper
    keeps the validator and the projector in agreement on which keys are
    normalisable.
    """
    if not isinstance(raw, dict):
        return raw
    return {
        k: ({} if (k in NORMALISABLE_BLOCK_KEYS and v is None) else v)
        for k, v in raw.items()
    }


class ConfigValidationError(ValueError):
    """Raised by ``validate_config(..., strict=True)`` on schema errors."""


def _format_typo_hints(exc: ValidationError) -> str:
    """Surface trailing-underscore typo hints from a pydantic ValidationError.

    The original validator was created to catch `agent.max_iterations_:`
    (stray underscore) — pydantic's generic `extra_forbidden` message
    only names the bad key, so we prepend a one-liner per offending key
    that names the likely correct spelling and the word "underscore"
    (so a grep over logs surfaces these for cleanup).
    """
    hints: list[str] = []
    for err in exc.errors():
        if err.get("type") != "extra_forbidden":
            continue
        loc = err.get("loc") or ()
        if not loc:
            continue
        bad_key = loc[-1]
        if not isinstance(bad_key, str) or not bad_key.endswith("_"):
            continue
        stripped = bad_key.rstrip("_")
        if not stripped:
            continue
        path = ".".join(str(p) for p in loc)
        hints.append(
            f"  - `{path}` has a trailing underscore; "
            f"did you mean `{stripped}`?"
        )
    if not hints:
        return ""
    return "Trailing-underscore typo(s) detected:\n" + "\n".join(hints) + "\n"


def validate_config(raw: dict, strict: bool = True) -> int:
    """Validate the raw YAML dict against the pydantic schema.

    Parameters
    ----------
    raw
        Parsed built-in config or YAML override mapping.
    strict
        If True (default), raise ``ConfigValidationError`` on the first
        validation error. If False, log a warning and return the count.

    Returns
    -------
    int
        Number of validation errors detected (always 0 when strict=True
        since we raise on the first one).
    """
    if not isinstance(raw, dict):
        raise ConfigValidationError(
            f"config root must be a mapping; got {type(raw).__name__}"
        )
    # Normalise null blocks to empty dicts so a commented-out section
    # (`agent: null` / `agent:` with no body) doesn't crash validation —
    # the pydantic models for these blocks each have a default factory.
    raw_for_validation = normalise_null_blocks(raw)
    try:
        AgentConfig.model_validate(raw_for_validation)
    except ValidationError as exc:
        typo_hint = _format_typo_hints(exc)
        fields = "\n".join(".".join(map(str, err["loc"])) + ": " + err["type"]
                           for err in exc.errors(include_input=False))
        msg = (
            "agent config has validation errors (typos or unknown keys "
            "are flagged as `extra forbidden`; fix them or set strict=False "
            f"to downgrade to a warning):\n{typo_hint}  {fields}"
        )
        if strict:
            raise ConfigValidationError(msg) from None
        logger.warning("config: %s", msg)
        return len(exc.errors())

    return 0
