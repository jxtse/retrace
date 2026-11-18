export const meta = {
  name: 'expval2-chemist-audit',
  description: 'Organic-chemist audit of all routes for the 3 ExpValidation2 cases; per-step mechanistic verdicts + per-case synthesized feedback for agent resume',
  phases: [
    { title: 'Audit', detail: 'one expert organic chemist per route; re-derive every disconnection step' },
    { title: 'Synthesize', detail: 'one synthesizer per case; merge route verdicts into actionable feedback for the agent' },
  ],
}

// args = the audit payload object: { "1": {idx,name,target,routes:[...]}, "2": {...}, "3": {...} }
const cases = args

const ROUTE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['overall_verdict', 'route_score_0to10', 'step_audits', 'global_issues', 'one_line_summary'],
  properties: {
    overall_verdict: { type: 'string', enum: ['VALID', 'SUSPECT', 'FAKE'],
      description: 'VALID=every step is a real, correctly-balanced, precedented disconnection. SUSPECT=at least one step is dubious/needs a specific reagent not implied. FAKE=at least one step is mechanistically impossible (wrong product, fabricated/lost skeletal carbon, written-target variant, impossible glycosylation/annulation, etc).' },
    route_score_0to10: { type: 'number', description: 'Holistic synthetic-chemist quality 0-10 (chemical feasibility, step economy, convergence, protecting-group sanity, real precedent).' },
    step_audits: {
      type: 'array',
      description: 'One entry per step, in the same order as given.',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['step_index', 'reaction_named', 'verdict', 'rationale'],
        properties: {
          step_index: { type: 'integer' },
          reaction_named: { type: 'string', description: 'What reaction the agent claims (from the annotation).' },
          verdict: { type: 'string', enum: ['OK', 'DUBIOUS', 'IMPOSSIBLE'] },
          rationale: { type: 'string', description: 'Concise mechanistic reasoning. If you re-derived product from precursors and it does NOT match the stated product, say so explicitly with the discrepancy (atom/group added or lost, wrong regiochemistry, wrong oxidation state).' },
          fix_suggestion: { type: 'string', description: 'If DUBIOUS/IMPOSSIBLE, the concrete chemistry fix (correct reagent, correct disconnection, missing protecting group, alternate sequence). Empty if OK.' },
        },
      },
    },
    global_issues: { type: 'array', items: { type: 'string' },
      description: 'Route-level (not step-level) problems: convergence, redundant steps, oxidation-state thrash, protecting-group strategy gaps, ordering hazards (e.g. acryloyl installed too early survives later harsh steps), buyability of claimed SM.' },
    one_line_summary: { type: 'string' },
  },
}

const CASE_FEEDBACK_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['case_idx', 'headline', 'best_route_label', 'feedback_to_agent', 'concrete_directives'],
  properties: {
    case_idx: { type: 'string' },
    headline: { type: 'string', description: 'One sentence: overall health of this case\'s route pool from a chemist\'s view.' },
    best_route_label: { type: 'string', description: 'Label of the route the chemist judges most synthetically sound (may differ from the agent\'s rank-1).' },
    feedback_to_agent: { type: 'string', description: 'The full chemist feedback message to hand back to the agent. Written AS a senior process chemist talking to the planning agent: what is wrong with specific steps (name the route label + step), what to fix, what to explore next. Faithful to the per-step audits. Must NOT invent praise; if the pool is weak, say so and direct re-planning.' },
    concrete_directives: { type: 'array', items: { type: 'string' },
      description: 'Bullet directives the agent can act on directly, e.g. "Replace step 4 SNAr of route X with ...", "Re-examine glycosylation in route Y; free-sugar donor is impossible, use an activated donor", "Save 2 more convergent routes avoiding early acryloylation".' },
  },
}

function renderRoute(c, r) {
  const steps = r.steps.map((s, i) => `  [step ${i}] ${s}`).join('\n')
  return `TARGET (${c.name}): ${c.target}
ROUTE label="${r.label}"  agent_rank=${r.rank}  agent_total_score=${r.total_score}  num_steps=${r.num_steps}
agent_sub_scores=${JSON.stringify(r.sub_scores)}
claimed_starting_materials=${JSON.stringify(r.starting_materials)}
STEPS (retrosynthetic order, each is  product >> precursors  (reaction annotation) ):
${steps}`
}

const AUDIT_PREAMBLE = `You are a senior process/organic chemist auditing one retrosynthetic route, step by step, from a global synthesis-design viewpoint.

CRITICAL METHOD — for EVERY step, do NOT trust the reaction-type label. Mentally run the reaction FORWARD: take the precursors (the right-hand side of ">>") and the named reaction, and derive what product they would actually give. Compare to the stated product (left-hand side). Flag a step IMPOSSIBLE when:
 - the forward reaction would NOT yield the stated product (extra or missing skeletal atom, wrong regiochemistry, wrong oxidation state, wrong connectivity);
 - a skeletal carbon is fabricated or lost that the reagents cannot account for (e.g. a H2O/NH3-only annulation that nonetheless changes the carbon count);
 - the stated product SMILES is merely a re-written copy of the TARGET that the declared reaction does not actually produce (written-target trap);
 - a bond construction is not chemically real with the given reagents (impossible glycosylation with a free sugar, impossible C–C coupling without a handle, etc.).
Flag DUBIOUS when the step is conceivable but needs a specific reagent/condition/protecting-group not implied, or has a serious selectivity problem.
Mark OK only when the disconnection is real, balanced on the main skeleton (mass balance on small molecules like H2O/HCl is NOT required — only the organic skeleton matters), and precedented.

Also judge the route GLOBALLY: convergence vs linearity, redundant or oxidation-state-thrashing steps, protecting-group sanity, and ORDERING hazards (e.g. an acryloyl/Michael acceptor installed early that would not survive later steps; a reactive group exposed too long). Buyability of the claimed starting materials is a plus but a leaf that is one trivial step from a catalog compound is fine.

Be a tough but fair reviewer. Return ONLY the structured verdict.`

phase('Audit')
// Pipeline: each route audits independently; as a case's routes finish they
// flow into that case's synthesizer. Build a flat list of {caseIdx, route}.
const flat = []
for (const idx of Object.keys(cases)) {
  const c = cases[idx]
  for (const r of c.routes) flat.push({ idx, c, r })
}
log(`Auditing ${flat.length} routes across ${Object.keys(cases).length} cases`)

const audits = await parallel(flat.map((item) => () =>
  agent(
    `${AUDIT_PREAMBLE}\n\n=== ROUTE TO AUDIT ===\n${renderRoute(item.c, item.r)}`,
    { label: `audit:c${item.idx}:${item.r.label.slice(0, 22)}`, phase: 'Audit', schema: ROUTE_SCHEMA }
  ).then((v) => ({ idx: item.idx, label: item.r.label, rank: item.r.rank,
      agent_score: item.r.total_score, audit: v }))
    .catch(() => null)
))

// Group audits by case.
const byCase = {}
for (const a of audits.filter(Boolean)) {
  (byCase[a.idx] = byCase[a.idx] || []).push(a)
}

phase('Synthesize')
const feedbacks = await parallel(Object.keys(cases).map((idx) => () => {
  const c = cases[idx]
  const ras = (byCase[idx] || []).sort((x, y) => x.rank - y.rank)
  const dossier = ras.map((a) =>
    `--- ROUTE "${a.label}" (agent_rank=${a.rank}, agent_score=${a.agent_score}) ---\n` +
    `chemist_verdict=${a.audit?.overall_verdict} chemist_score=${a.audit?.route_score_0to10}/10\n` +
    `summary: ${a.audit?.one_line_summary}\n` +
    `step_audits:\n${(a.audit?.step_audits || []).map((s) =>
      `  step ${s.step_index} [${s.verdict}] (${s.reaction_named}): ${s.rationale}` +
      (s.fix_suggestion ? `  FIX: ${s.fix_suggestion}` : '')).join('\n')}\n` +
    `global_issues: ${JSON.stringify(a.audit?.global_issues || [])}`
  ).join('\n\n')
  return agent(
    `You are the lead chemist consolidating a per-route audit dossier for ONE target into a single feedback message for an autonomous retrosynthesis planning agent that will RESUME its search after reading your feedback.

TARGET: ${c.name} = ${c.target}
The agent saved ${c.n_routes} routes. Below are the per-route chemist audits.

${dossier}

Write feedback that will make the agent's NEXT iteration produce better chemistry. Be specific: cite route labels and step indices. Where a step is IMPOSSIBLE/DUBIOUS, tell the agent exactly what is wrong and what the correct disconnection or reagent is. If the whole pool shares a systemic flaw (e.g. all routes install the acrylamide too early, or all rely on the same impossible coupling), say so and direct a different strategy. If some routes are genuinely sound, tell the agent to keep/refine those and stop wasting budget re-deriving them. Do not flatter. Return the structured feedback.`,
    { label: `synth:c${idx}`, phase: 'Synthesize', schema: CASE_FEEDBACK_SCHEMA }
  ).then((v) => ({ idx, n_routes_audited: ras.length, feedback: v })).catch(() => null)
}))

return {
  audited_routes: audits.filter(Boolean).length,
  per_route_audits: audits.filter(Boolean),
  per_case_feedback: feedbacks.filter(Boolean),
}
