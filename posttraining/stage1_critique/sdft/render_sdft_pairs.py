"""Native-template data formatting only; no generation or training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def normalize_messages(messages):
    """Decode OpenAI JSON argument strings for HF's native chat-template API."""
    result = copy.deepcopy(messages)
    def object_no_duplicates(pairs):
        obj = {}
        for k, v in pairs:
            if k in obj:
                raise ValueError('Duplicate tool-argument key')
            obj[k] = v
        return obj
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON tool argument: '+value)
    for message in result:
        for call in message.get('tool_calls', []):
            function = call['function']
            value = function['arguments']
            if isinstance(value, str):
                value = json.loads(value, object_pairs_hook=object_no_duplicates, parse_constant=invalid_constant)
            if not isinstance(value, dict):
                raise ValueError('Tool arguments must be a JSON object')
            function['arguments'] = value
    return result


def render_pair(messages, tools, reference, tokenizer):
    if not tools or not reference or not isinstance(reference, str):
        raise ValueError('Tool schemas and privileged reference are required')
    if not messages or any(m.get('role') not in {'system', 'user', 'assistant', 'tool'} for m in messages):
        raise ValueError('Expected native chat messages, not flattened trajectory roles')
    messages = normalize_messages(messages)
    teacher = copy.deepcopy(messages)
    user = next((m for m in teacher if m['role'] == 'user'), None)
    if user is None or not isinstance(user.get('content'), str):
        raise ValueError('Missing textual task request')
    user['content'] += ('\n\n<privileged_reference>\nReference labeled VALID by the frozen audit, '
                        'not experimental chemical truth. Use it to inform the original task.\n'
                        + reference + '\n</privileged_reference>')
    def render(value):
        return tokenizer.apply_chat_template(value, tools=tools, tokenize=False, add_generation_prompt=True)
    return {'prompt': render(messages), 'privileged_context': render(teacher)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--references', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from types import SimpleNamespace
    from trl.experimental.sdft.sdft_trainer import DemonstrationTeacherContextBuilder
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='left')
    refs = {x['idx']: x for x in map(json.loads, Path(args.references).read_text().splitlines())}
    rows, selected = [], set()
    for line in Path(args.source).open():
        source = json.loads(line)
        idx = source['case_idx']
        if idx in selected or idx not in refs:
            continue
        reference = refs[idx]
        assert reference['route_verdict'] == 'VALID'
        assert source['target_smiles'] == reference['target_smiles']
        # Only the genuine initial task request is used here. Later legacy
        # flattened tool_call roles must be read from native llm_requests instead.
        messages = source['messages'][:2]
        assert [m['role'] for m in messages] == ['system','user']
        route = '\n'.join(s['raw'] for s in reference['steps'])
        row = render_pair(messages, source['tools'], route, tok)
        row['meta'] = {'idx': idx, 'reference_uid': reference['uid'], 'tools': len(source['tools']),
                       'scope': 'native-tools initial-prompt capacity probe, not a multi-turn rollout'}
        row['meta']['prompt_tokens_student_teacher'] = [
            len(tok(row[k], add_special_tokens=False)['input_ids']) for k in ['prompt','privileged_context']]
        # Verify the exact installed trainer preserves the pre-rendered teacher.
        builder = DemonstrationTeacherContextBuilder(SimpleNamespace(
            args=SimpleNamespace(teacher_prompt_template='{privileged_context}')))
        assert builder._compose_teacher_prompt(row['prompt'], row['privileged_context']) == row['privileged_context']
        assert '<privileged_reference>' not in row['prompt']
        rows.append(row)
        selected.add(idx)
        if len(rows) == 4:
            break
    assert len(rows) == 4
    output = Path(args.output)
    with output.open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    print(json.dumps({'rows': [x['meta'] for x in rows],
                      'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                      'sources': {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                  for p in [args.source,args.references]}}, indent=2))


if __name__ == '__main__':
    main()
