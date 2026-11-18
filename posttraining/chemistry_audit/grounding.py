"""Verbatim function extraction; see SOURCE_PINS.json. No source main is run."""
import re
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")


def canon(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None

def carbon_count(s):
    m = Chem.MolFromSmiles(s)
    if not m:
        return None
    return sum(1 for a in m.GetAtoms() if a.GetSymbol() == "C")

def ring_sizes(s):
    m = Chem.MolFromSmiles(s)
    if not m:
        return None
    ri = m.GetRingInfo()
    return sorted(len(r) for r in ri.AtomRings())

def parse_step(raw):
    """`product>>r1.r2 (reaction: name)` -> (product, [reactants], name)."""
    name = ""
    mm = re.search(r"\(reaction:\s*(.*?)\)\s*$", raw)
    if mm:
        name = mm.group(1).strip()
        raw = raw[: mm.start()].strip()
    if ">>" not in raw:
        return None, [], name
    prod, reac = raw.split(">>", 1)
    prod = prod.strip()
    reactants = [x.strip() for x in reac.split(".") if x.strip()]
    return prod, reactants, name

def ground_step(i, raw):
    prod, reactants, name = parse_step(raw)
    pc = carbon_count(prod) if prod else None
    rc_list = [carbon_count(r) for r in reactants]
    parse_ok = prod is not None and all(carbon_count(r) is not None for r in reactants) and pc is not None
    cdelta = None
    if parse_ok:
        cdelta = pc - sum(rc_list)  # product_C - sum precursor_C
    rs_prod = ring_sizes(prod) if prod else None
    rs_reac = []
    for r in reactants:
        rs = ring_sizes(r)
        if rs:
            rs_reac.extend(rs)
    rs_reac = sorted(rs_reac)
    ring_delta = None
    if rs_prod is not None:
        # multiset difference summary: rings gained in product vs all reactants
        from collections import Counter
        gained = Counter(rs_prod) - Counter(rs_reac)
        lost = Counter(rs_reac) - Counter(rs_prod)
        ring_delta = {"gained": sorted(gained.elements()), "lost": sorted(lost.elements())}
    return {
        "i": i,
        "raw": raw,
        "product": prod,
        "reactants": reactants,
        "reaction_name": name,
        "parse_ok": parse_ok,
        "carbon_delta": cdelta,
        "ring_sizes_product": rs_prod,
        "ring_sizes_reactants": rs_reac,
        "ring_delta": ring_delta,
    }
