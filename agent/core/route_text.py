"""Helpers for parsing route text emitted by the agent."""

from __future__ import annotations


def parse_route_from_text(route_text: str) -> list[dict]:
    """
    Parse the agent's free-form COMPLETE ROUTE FOUND text into structured steps.

    Strategy: first extract every backtick-delimited chunk that contains '>>'
    and treat each as one reaction. This is robust to bolding, spacing,
    numbering format, and SMILES that contain parentheses.

    Returns a list of {"product": SMILES, "reactants": "R1.R2"} dicts.
    """
    import re as _re
    if not route_text:
        return []

    # Crop to the COMPLETE ROUTE section
    m_hdr = _re.search(r"COMPLETE\s+ROUTE\s+FOUND", route_text, _re.IGNORECASE)
    body = route_text[m_hdr.end():] if m_hdr else route_text

    steps: list[dict] = []
    seen = set()

    # Pattern A: a single backtick block containing >>, e.g. `A >> B.C`
    # (the most common format for aspirin-style single steps)
    for m in _re.finditer(r"`([^`\n]*>>[^`\n]*)`", body):
        chunk = m.group(1)
        if ">>" not in chunk:
            continue
        product, _, reactants_blob = chunk.partition(">>")
        product = product.strip().strip("`").strip()
        reactants_blob = reactants_blob.strip().strip("`").strip()
        parts = [r.strip() for r in reactants_blob.split(".") if r.strip()]
        reactants = ".".join(parts)
        key = (product, reactants)
        if product and reactants and key not in seen:
            seen.add(key)
            steps.append({"product": product, "reactants": reactants})

    # Pattern B: backticked product, >>, backticked reactant(s) joined by .
    # e.g.  `PRODUCT` >> `R1`.`R2`
    # Greedy across backticks around the >> arrow.
    pattern_b = _re.compile(
        r"`([^`\n]+?)`\s*>>\s*((?:`[^`\n]+?`(?:\s*\.\s*)?)+)"
    )
    for m in pattern_b.finditer(body):
        product = m.group(1).strip()
        reactants_seg = m.group(2)
        reactant_list = _re.findall(r"`([^`\n]+?)`", reactants_seg)
        reactant_list = [r.strip() for r in reactant_list if r.strip()]
        reactants = ".".join(reactant_list)
        key = (product, reactants)
        if product and reactants and key not in seen:
            seen.add(key)
            steps.append({"product": product, "reactants": reactants})

    # Pattern C: plain `Step N: PRODUCT >> R1.R2 (reaction: ...)` line — the
    # exact format the prompt template (prompts/core.md) instructs the LLM to
    # emit. Without this branch the cheat detector and buyability gate run
    # on an empty `route_steps` list and never fire (case 6 dronedarone
    # slipped through as a single-step cheat for exactly this reason).
    pattern_c = _re.compile(
        r"(?im)^\s*Step\s+\d+\s*:\s*(?P<prod>\S.*?)\s*>>\s*(?P<reacts>.+?)\s*(?:\(reaction:[^)]*\))?\s*$"
    )
    for m in pattern_c.finditer(body):
        product = m.group("prod").strip().strip("`").strip()
        reactants_blob = m.group("reacts").strip().strip("`").strip()
        parts = [r.strip().strip("`").strip() for r in reactants_blob.split(".") if r.strip()]
        reactants = ".".join(parts)
        key = (product, reactants)
        if product and reactants and key not in seen:
            seen.add(key)
            steps.append({"product": product, "reactants": reactants})

    # Order by Step N if explicit numbering is present; otherwise keep insertion order.
    step_positions: list[tuple[int, dict]] = []
    for st in steps:
        # Find the earliest Step N header that appears before this step's product
        # in the body text, to use as sort key.
        prod_pos = body.find(st["product"])
        best_step_num = 10**6  # default huge so unknown steps sink
        for sm in _re.finditer(r"Step\s*(\d+)", body, _re.IGNORECASE):
            if sm.start() <= prod_pos:
                best_step_num = int(sm.group(1))
        step_positions.append((best_step_num, st))
    step_positions.sort(key=lambda kv: kv[0])
    return [kv[1] for kv in step_positions]
