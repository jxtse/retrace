"""Versioned S3 admission boundary. No model dependencies or scientific scoring."""
import hashlib
import json
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
SCHEMA = 'retrace.s3.final-selected.v3'
SPLIT_SHA = 'ec256d661c77017befa2235e5c2595f46bac226256490562188fa2337991c3a9'
PREDECESSOR_SHA = '370edde7b44e0c4ef3cedcc35492306e1a95773a67fafd65f6ef4e6c53405e02'
CONFIG_SHA = '8585a420b0f4dd62cd2a79d09e39173e216b3f85e62794df854ff25c689c0861'
OWNER_POLICY = 'OWNER_AUTHORIZED_INTERNAL_RUN'
OWNER_INDEX_SHA = '7a1567a3765e8797427b00214f859d79474539cbe529a48e154c3975c58d9d95'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


def verify_ref(root, ref):
    require(set(ref) == {'path', 'sha256'}, 'hash reference needs path and sha256')
    p = Path(ref['path'])
    p = p if p.is_absolute() else Path(root) / p
    require(isinstance(ref['sha256'], str) and len(ref['sha256']) == 64, 'invalid digest')
    require(p.is_file() and sha(p) == ref['sha256'], 'hash mismatch: ' + str(p))
    return p


def verify_policy_ref(root, ref):
    """Policy evidence must travel inside its bundle, with no absolute/escaping refs."""
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'policy reference schema')
    p = Path(ref['path'])
    require(not p.is_absolute() and '..' not in p.parts
            and (Path(root) / p).resolve().is_relative_to(Path(root).resolve()), 'nonportable policy reference')
    return verify_ref(root, ref)


def validate_owner_decision(decision, source_keys, selected_index_sha256):
    """Execution authorization is not a claim of legal or provider permission."""
    required = {'schema', 'data_origin', 'status', 'legal_clearance', 'execution_authorized',
                'no_public_release', 'selected_index_sha256', 'scope_source_keys', 'artifacts'}
    require(isinstance(decision, dict) and required <= set(decision)
            and set(decision) <= required | {'rights_verified', 'training_host', 'initialization_sha256', 'created_at'},
            'owner decision fields/claims invalid')
    require(decision['schema'] == 'retrace.s3.owner-execution-decision.v1'
            and decision['data_origin'] == 'actual_s3' and decision['status'] == OWNER_POLICY,
            'explicit owner decision required')
    require(decision['legal_clearance'] == 'UNVERIFIED' and decision.get('rights_verified', False) is False,
            'owner authorization cannot verify legal rights')
    require(decision['execution_authorized'] is True and decision['no_public_release'] is True,
            'internal execution authorization required')
    require(decision['selected_index_sha256'] == selected_index_sha256 == OWNER_INDEX_SHA,
            'owner selected index mismatch')
    keys = decision['scope_source_keys']
    require(isinstance(keys, list) and all(isinstance(k, str) for k in keys)
            and len(keys) == len(set(keys)) and set(keys) == set(source_keys), 'owner source scope mismatch')
    require(decision.get('training_host', RETRACE_HOST) == RETRACE_HOST
            and decision.get('initialization_sha256', PREDECESSOR_SHA) == PREDECESSOR_SHA,
            'owner host/initialization mismatch')
    require(isinstance(decision['artifacts'], list) and bool(decision['artifacts']), 'owner evidence missing')


def verify_owner_policy(root, gate, frozen_sources, source_keys):
    require(gate.get('execution_policy') == OWNER_POLICY and gate.get('rights_verified') is False
            and gate.get('execution_authorized') is True and gate.get('legal_clearance') == 'UNVERIFIED'
            and gate.get('no_public_release') is True, 'owner execution policy missing or legal claim')
    require(not {'legal_verified', 'provider_permission', 'rights_evidence'} & set(gate), 'owner legal claims forbidden')
    index_ref = frozen_sources['selected_index']
    require(index_ref['sha256'] == OWNER_INDEX_SHA, 'owner selected index mismatch')
    index = verify_ref(root, index_ref)
    indexed_keys = [json.loads(line)['source_key'] for line in index.read_text().splitlines() if line.strip()]
    require(len(indexed_keys) == len(source_keys) and set(indexed_keys) == set(source_keys), 'owner index source mismatch')
    path = verify_policy_ref(root, gate['owner_decision'])
    decision = read_json(path)
    validate_owner_decision(decision, source_keys, index_ref['sha256'])
    for ref in decision['artifacts']:
        verify_policy_ref(path.parent, ref)
    verify_policy_ref(root, gate['exporter_source'])


def validate_tokens(row, think_ids=None, positive=False):
    ids, labels = row['input_ids'], row['labels']
    eligible, selected = row['eligible_mask'], row['selected_mask']
    require(type(ids) is list and 0 < len(ids) <= 131072, 'full context outside cap')
    require(all(type(x) is int and x >= 0 for x in ids), 'integer IDs required')
    for mask in (eligible, selected):
        require(type(mask) is list and len(mask) == len(ids), 'mask length mismatch')
        require(all(type(x) is int and x in (0, 1) for x in mask), 'binary integer mask required')
    require(type(labels) is list and all(type(x) is int for x in labels), 'integer labels required')
    require(labels == [x if s else -100 for x, s in zip(ids, selected)], 'labels not exact selected mask')
    require(all(s <= e for s, e in zip(selected, eligible)), 'selected exceeds eligibility')
    require(eligible[0] == selected[0] == 0, 'first causal token must be ignored')
    if positive:
        require(sum(selected) > 0, 'zero-loss row is not trainable')
    if think_ids is not None:
        opening, closing = think_ids
        require(opening != closing and opening >= 0 and closing >= 0, 'invalid thinking token IDs')
        inside = False
        for token, e in zip(ids, eligible):
            if token == opening:
                inside = True
            if inside or token == closing:
                require(e == 0, 'reasoning eligibility forbidden')
            if token == closing:
                inside = False
        require(not inside, 'unterminated thinking span')
    return sum(selected)


class Release:
    """Require an independently supplied manifest SHA; never infer trust from a Boolean."""
    def __init__(self, path, expected_sha):
        self.path = Path(path).resolve()
        require(bool(expected_sha) and sha(self.path) == expected_sha, 'manifest digest mismatch')
        self.sha256 = expected_sha
        self.root = self.path.parent
        self.manifest = m = read_json(self.path)
        require(m['schema'] == SCHEMA and m['mode'] == 'FINAL_SELECTED', 'FINAL_SELECTED only')
        require(m['training_ready'] is True and m['data_origin'] == 'actual_s3', 'data gate false or fixture')
        require(m['predecessor_sha256'] == PREDECESSOR_SHA, 'wrong predecessor')
        require(m['split']['sha256'] == SPLIT_SHA, 'wrong frozen split')
        split = read_json(verify_ref(self.root, m['split']))
        # Exact installed frozen split schema.
        train, held = set(split['train_stage2']), set(split['dev']) | set(split['test40_eval_only'])
        require(not train & held, 'split overlap')
        require(m['rows'] and len({r['uid'] for r in m['rows']}) == len(m['rows']), 'empty/duplicate rows')
        require(len({r['source_key'] for r in m['rows']}) == len(m['rows']), 'duplicate source keys')
        gate = read_json(verify_ref(self.root, m['parent_gate']))
        require(gate['stage'] == 'SCIENTIFIC_SELECTED' and gate['training_ready'] is True,
                'parent scientific gate missing')
        require(gate['data_origin'] == 'actual_s3' and gate['rows_sha256'] == digest(m['rows']), 'gate row binding mismatch')
        require(gate['all_causal_prefixes_verified'] is True, 'source provenance gate missing')
        if 'owner_decision' in gate or gate.get('execution_policy') == OWNER_POLICY:
            verify_owner_policy(self.root, gate, m['frozen_sources'], [r['source_key'] for r in m['rows']])
        else:
            require(gate['rights_verified'] is True, 'source provenance/rights gate missing')
        for ref in m['frozen_sources'].values():
            verify_ref(self.root, ref)
        require(m['frozen_sources'], 'frozen source pins missing')
        self.rows = {r['uid']: r for r in m['rows']}
        self.total_tokens = self.selected_tokens = 0
        for r in m['rows']:
            require(type(r['target_id']) is int and r['target_id'] in train and r['target_id'] not in held,
                    'held-out or unknown target')
            require(r['source_key'] not in {'174_rep2', '069_rep3', '160_rep3', '011_rep4'}, 'quarantined source')
            require(r['data_origin'] == 'actual_s3', 'fixture row forbidden')
            require(set(r['sources']) >= {'result', 'messages', 'requests', 'transcript', 'tools'}, 'incomplete source chain')
            for ref in r['sources'].values():
                verify_ref(self.root, ref)
            p = self.payload(r['uid'])
            self.selected_tokens += validate_tokens(p, m['think_token_ids'], positive=True)
            self.total_tokens += len(p['input_ids'])
            receipts = r['receipts']
            require(set(receipts) == {'audit', 'selector1', 'selector2', 'admission'}, 'receipt set mismatch')
            require(receipts['selector1']['sha256'] != receipts['selector2']['sha256'], 'selector samples must be distinct receipts')
            for ref in receipts.values():
                verify_ref(self.root, ref)
            # The admission coder normalizes/verifies frozen scientific output. This reader
            # validates its binding, not chemistry or generated scores.
            a = read_json(verify_ref(self.root, receipts['admission']))
            require(a['status'] == 'FINAL_SELECTED' and a['data_origin'] == 'actual_s3', 'not final actual admission')
            require(a['source_key'] == r['source_key'] and a['payload_sha256'] == r['payload']['sha256'], 'admission payload mismatch')
            require(a['sources'] == r['sources'], 'admission source mismatch')
            require(a['audit_sha256'] == receipts['audit']['sha256'] and a['route_verdict'] == 'VALID', 'audit binding/VALID missing')
            require(a['selector_sha256'] == [receipts['selector1']['sha256'], receipts['selector2']['sha256']], 'two-selector binding mismatch')
            require(a['selected_mask_sha256'] == digest(p['selected_mask']), 'selection mask digest mismatch')
            for key in ('native_success', 'screen_accepted', 'causal_prefixes_verified', 'mechanical_veto_applied', 'frozen_rule_verified'):
                require(a[key] is True, 'admission missing: ' + key)
        require(m['counts'] == {'rows': len(self.rows), 'full_tokens': self.total_tokens, 'selected_tokens': self.selected_tokens}, 'exposure counts mismatch')

    def payload(self, uid):
        require(uid in self.rows, 'unknown uid')
        r = self.rows[uid]
        p = read_json(verify_ref(self.root, r['payload']))
        require(p['uid'] == uid and p['source_key'] == r['source_key'], 'payload identity mismatch')
        return p

    def record(self, uid):
        p = self.payload(uid)
        return dict(uid=uid, mode='FINAL_SELECTED', input_ids=p['input_ids'], labels=p['labels'], lengths=[len(p['input_ids'])])
