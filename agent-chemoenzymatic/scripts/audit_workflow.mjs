export const meta = {
  name: 'retrace-chemist-audit',
  description: 'Chemist-grade per-step audit of every top-10 RETRACE route (standard+fast), adversarial verify of FAKE verdicts',
  whenToUse: 'Auditing retro-190 RETRACE route quality at top-1/3/5/10 depth',
  phases: [
    { title: 'Audit', detail: 'one expert-chemist agent per route; per-step VALID/SUSPECT/FAKE + route verdict' },
    { title: 'Verify', detail: 'adversarial refutation of every route the auditor called FAKE' },
  ],
}

// args = { uids: ["standard:0:r1", ...], payloadFile: "/tmp/audit_payloads.json" }
// Each agent fetches its OWN route record from payloadFile by uid via Bash —
// args stays tiny and one agent's fetch failure can't zero the run.
const UIDS = (args && args.uids) || []
const PF = (args && args.payloadFile) || '/tmp/audit_payloads.json'
log(`auditing ${UIDS.length} routes (fetch-by-uid from ${PF})`)

const STEP_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['step_index', 'verdict', 'defect_code', 'reason'],
  properties: {
    step_index: { type: 'integer' },
    verdict: { enum: ['VALID', 'SUSPECT', 'FAKE'] },
    defect_code: {
      enum: ['NONE', 'TARGET_NOT_PRODUCED', 'WRONG_RING_SIZE',
             'FABRICATED_OR_MISSING_SKELETAL_CARBON', 'WRONG_REGIO_OR_CONSTITUTIONAL',
             'IMPOSSIBLE_OR_NONSENSE_REACTION', 'WRONG_REAGENT_OR_CONDITIONS',
             'STEREO_ONLY', 'OTHER'],
    },
    reason: { type: 'string' },
  },
}
const AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['uid', 'route_verdict', 'worst_defect_code', 'n_fake_steps', 'n_suspect_steps', 'steps', 'summary'],
  properties: {
    uid: { type: 'string' },
    route_verdict: { enum: ['VALID', 'SUSPECT', 'FAKE'] },
    worst_defect_code: { type: 'string' },
    n_fake_steps: { type: 'integer' },
    n_suspect_steps: { type: 'integer' },
    steps: { type: 'array', items: STEP_SCHEMA },
    summary: { type: 'string' },
  },
}
const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['uid', 'fake_upheld', 'corrected_verdict', 'reason'],
  properties: {
    uid: { type: 'string' },
    fake_upheld: { type: 'boolean' },
    corrected_verdict: { enum: ['VALID', 'SUSPECT', 'FAKE'] },
    reason: { type: 'string' },
  },
}

function auditPrompt(uid) {
  return `You are a senior synthetic/medicinal chemist auditing ONE retrosynthetic route for scientific validity.

FIRST, fetch the route. Run exactly this Bash command and read its stdout (a JSON object):
  python3 -c "import json; d=[r for r in json.load(open('${PF}')) if r['uid']=='${uid}'][0]; print(json.dumps(d))"

The JSON has: target (SMILES), steps (list of "reaction_name: PRODUCT  <<  PRECURSOR1.PRECURSOR2   [RDKit tags]"), flags{target_produced, any_unparseable, n_dangling, dangling}.

Judge each step as a real organic/enzymatic reaction converting the listed PRECURSORS into the listed PRODUCT. The route may be written target-first or building-block-first; reason about each step independently as precursors→product. Per-step "[Cdelta=+N]" is product_carbons − Σprecursor_carbons (RDKit). "ringΔ" lists ring sizes gained/lost in the product vs precursors (RDKit). flags are deterministic RDKit facts — TRUST them: target_produced=false means the target is never synthesized (hard defect); n_dangling>0 means a reactant is neither produced by another step nor a starting material.

For EACH step decide VALID / SUSPECT / FAKE:
- FAKE = the named reaction cannot give the stated product from the stated precursors. Triggers:
  * IMPOSSIBLE_OR_NONSENSE_REACTION: no known mechanism does this transformation.
  * FABRICATED_OR_MISSING_SKELETAL_CARBON: skeletal carbons appear/vanish with no carbon-bearing reagent to account (use Cdelta — but a nonzero Cdelta is FINE if a reagent legitimately supplies/removes that carbon, e.g. formaldehyde, CO2 loss, succinimide/acetate leaving as a unit, protecting-group reagents).
  * WRONG_RING_SIZE: a cycloaddition/annulation yields a ring of the wrong size (Diels–Alder of a C5 diene + C2 dienophile must give a 6-membered ring with 100% atom economy; check ringΔ and Cdelta).
  * WRONG_REGIO_OR_CONSTITUTIONAL: product is a constitutional/regio isomer of what the reaction would actually give.
  * WRONG_REAGENT_OR_CONDITIONS: the reagent class cannot effect the change (e.g. NBS displacing an acetate; DCM as a chloromethylation source).
  * TARGET_NOT_PRODUCED: when flags.target_produced=false.
- SUSPECT = plausible but a real concern (low-yield, chemoselectivity not obvious, unusual but not impossible). STEREO_ONLY for stereochemistry-only doubts.
- VALID = a competent chemist would accept this step.

A carbon-CONSERVED step can still be FAKE (carbon conservation does NOT prove mechanism). Be skeptical but fair: do not fail a step merely for being enzymatic, protecting-group-heavy, or low-yield.

route_verdict = FAKE if ANY step is FAKE; SUSPECT if any step SUSPECT but none FAKE; else VALID. worst_defect_code = the most serious step defect_code (NONE if VALID). Return uid="${uid}". Your structured output IS the result.`
}

function verifyPrompt(uid, audit) {
  const fakeSteps = (audit.steps || []).filter(s => s.verdict === 'FAKE')
    .map(s => `  step[${s.step_index}] flagged ${s.defect_code}: ${s.reason}`).join('\n')
  return `You are a skeptical chemist asked to REFUTE a FAKE verdict — steelman the route. Another auditor called route ${uid} FAKE.

FIRST fetch the route:
  python3 -c "import json; d=[r for r in json.load(open('${PF}')) if r['uid']=='${uid}'][0]; print(json.dumps(d))"

FLAGGED AS FAKE by the first auditor:
${fakeSteps}

Find the strongest case that the flagged step(s) are achievable real chemistry (allowing enzymatic/biocatalytic, multi-component, or non-obvious but literature-precedented transformations). Decide honestly: does the FAKE verdict hold up (fake_upheld=true) or can the step(s) be defended (fake_upheld=false)? Default fake_upheld=true ONLY if you genuinely cannot defend the chemistry. corrected_verdict = your final call for the whole route. Return uid="${uid}".`
}

phase('Audit')
const results = await pipeline(
  UIDS,
  (uid) => agent(auditPrompt(uid), { label: `audit ${uid}`, phase: 'Audit', schema: AUDIT_SCHEMA })
            .then(a => ({ uid, a }))
            .catch(() => ({ uid, a: null })),
  ({ uid, a }) => {
    const [cond, idx, rtag] = uid.split(':')
    const rank = parseInt((rtag || 'r0').slice(1), 10)
    if (!a) return { uid, cond, idx: parseInt(idx, 10), rank, route_verdict: 'ERROR', worst_defect_code: 'ERROR', verified: false }
    const base = { uid, cond, idx: parseInt(idx, 10), rank,
                   route_verdict: a.route_verdict, worst_defect_code: a.worst_defect_code,
                   n_fake_steps: a.n_fake_steps, n_suspect_steps: a.n_suspect_steps, summary: a.summary }
    if (a.route_verdict !== 'FAKE') return { ...base, verified: false }
    return agent(verifyPrompt(uid, a), { label: `verify ${uid}`, phase: 'Verify', schema: VERIFY_SCHEMA })
      .then(v => ({ ...base, verified: true, fake_upheld: v.fake_upheld,
                    route_verdict: v.corrected_verdict, verify_reason: v.reason }))
      .catch(() => ({ ...base, verified: false, verify_failed: true }))
  }
)

return results.filter(Boolean)
