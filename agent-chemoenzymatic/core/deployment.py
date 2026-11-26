"""Resolve deployment settings without constructing clients or reading datasets."""
import os
from pathlib import Path
from urllib.parse import urlsplit


def stock_path(base_dir, value):
    path = Path(value).expanduser()
    resolved = Path(base_dir) / path
    if path.is_absolute() or resolved.exists():
        return resolved
    legacy = Path(base_dir) / "vendor" / "legacy_retro_planner" / path
    return legacy if legacy.exists() else resolved


def retroplanner_urls(neural):
    for field in ("RETROPLANNER_HTTP_URLS", "RETROPLANNER_HTTP_URL"):
        value = os.environ.get(field)
        if value:
            urls = value.replace(",", " ").split()
            if field.endswith("URL") and len(urls) == 1:
                return urls[0]
            return urls
    return neural.get("retroplanner_http_urls") or neural.get("retroplanner_http_url")


ENGINE_ENDPOINTS = {
    "chembart": ("chembart_url", "CHEMBART_API_URL"),
    "pistachio_ringbreaker": ("pistachio_ringbreaker_url", "RINGBREAKER_API_URL"),
}


def engine_url(neural, engine):
    field, variable = ENGINE_ENDPOINTS[engine]
    return os.environ.get(variable) or neural.get(field)


def _valid_url(value):
    if not isinstance(value, str) or not value.isprintable() or any(c.isspace() for c in value):
        return False
    try:
        parts = urlsplit(value)
        _ = parts.port  # Accessing .port also range-checks it.
        return (parts.scheme in {"http", "https"} and bool(parts.hostname)
                and parts.username is None and not parts.query and not parts.fragment
                and not any(c in value for c in ("\\", ",")))
    except ValueError:
        return False


def shared_engines(neural):
    engines = set(neural.get("engines", ["graph_retrosyn", "rxngraphormer", *ENGINE_ENDPOINTS]))
    return (engines & {"graph_retrosyn", "rxngraphormer"}) | {
        engine for engine in engines & ENGINE_ENDPOINTS.keys() if not engine_url(neural, engine)
    }


def _block(parent, field, path):
    value = parent.get(field, {})
    if not isinstance(value, dict):
        raise ValueError(f"{path} requires a YAML mapping")
    return value


def _enabled(block, path, default):
    enabled = block.get("enabled", default)
    if not isinstance(enabled, bool):
        raise ValueError(f"{path}.enabled requires a YAML boolean")
    return enabled


def validate_chemprice(verifier):
    if verifier.get("cost") is None:
        return
    cost = _block(verifier, "cost", "verifier.cost")
    if cost.get("chemprice") is None:
        return
    pricing = _block(cost, "chemprice", "verifier.cost.chemprice")
    _enabled(pricing, "verifier.cost.chemprice", True)


def validate_deployment(config):
    """Validate effective destinations for the selected capabilities."""
    errors = []

    def endpoint(field, value, multiple=False):
        values = value if multiple and isinstance(value, list) else [value]
        if not values or not all(_valid_url(url) for url in values):
            errors.append(field + " requires an HTTP(S) URL without credentials")

    llm = config.get("llm", {})
    primary = config.get("llm_base_url", llm.get("base_url"))
    endpoint("llm.base_url", primary)
    if not (config.get("llm_model", llm.get("model")) or "").strip():
        errors.append("llm.model requires a served model ID")

    post = config.get("postprocess", {})
    clients = [("llm", llm)]
    if post.get("enabled"):
        clients.append(("postprocess", post))
        if post.get("base_url"):
            endpoint("postprocess.base_url", post["base_url"])
    for name, client in clients:
        fallback = client.get("fallback", {})
        if fallback.get("enabled"):
            endpoint(name + ".fallback.base_url", fallback.get("base_url", "https://openrouter.ai/api/v1"))

    verifier = config.get("verifier", {})
    if _enabled(verifier, "verifier", True):
        validate_chemprice(verifier)
        feas = _block(verifier, "feasibility", "verifier.feasibility")
        rater = _block(feas, "rater", "verifier.feasibility.rater")
        value = rater.get("http_url")
        endpoint("verifier.feasibility.rater.http_url", os.getenv("REACTION_RATER_URL") or value, True)
        judge = _block(feas, "judge", "verifier.feasibility.judge")
        if _enabled(judge, "verifier.feasibility.judge", True) and judge.get("base_url"):
            endpoint("verifier.feasibility.judge.base_url", judge["base_url"])

    neural = config.get("neural_retro", {})
    if neural.get("enabled", False):
        engines = set(neural.get("engines", ["graph_retrosyn", "rxngraphormer", *ENGINE_ENDPOINTS]))
        if any(mode != "http_only" for mode in neural.get("engine_modes", {}).values()):
            errors.append("neural_retro.engine_modes supports only http_only")
        if engines - {"graph_retrosyn", "rxngraphormer", "template_relevance", *ENGINE_ENDPOINTS}:
            errors.append("neural_retro.engines contains an unsupported engine")
        shared = retroplanner_urls(neural)
        needs_shared = shared_engines(neural)
        for engine, (field, _) in ENGINE_ENDPOINTS.items():
            if engine in engines:
                value = engine_url(neural, engine)
                if value:
                    endpoint("neural_retro." + field, value)
        if needs_shared:
            endpoint("neural_retro.retroplanner_http_url / neural_retro.retroplanner_http_urls", shared, True)
        if "template_relevance" in engines:
            endpoint("neural_retro.template_relevance_url", neural.get("template_relevance_url"))
        sanity = neural.get("precursor_sanity_filter", {})
        if _enabled(sanity, "neural_retro.precursor_sanity_filter", True):
            judge = _block(sanity, "judge_llm", "neural_retro.precursor_sanity_filter.judge_llm")
            endpoint("neural_retro.precursor_sanity_filter.judge_llm.base_url",
                     os.getenv("SANITY_JUDGE_BASE_URL") or judge.get("base_url") or primary)
    if errors:
        raise ValueError("\n".join(errors))
