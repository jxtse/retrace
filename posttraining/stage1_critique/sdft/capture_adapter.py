"""Data-only native decision-state conversion; no live/training admission authority."""

import argparse
import copy
import hashlib
import json
from pathlib import Path


MANIFEST_SHA256 = '37144b9dbe65d8bd04768ef4e1b7e9c5de2cd522beb29a7865a1dfbd60f5c929'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def load_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'Duplicate JSON key: {key}')
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f'Nonfinite JSON: {value}')

    result = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    encoded(result)
    return result


def normalize_messages(messages):
    result = copy.deepcopy(messages)
    require(isinstance(result, list), 'Messages must be a native list')
    for message in result:
        require(isinstance(message, dict), 'Message must be an object')
        calls = message.get('tool_calls', [])
        require(isinstance(calls, list), 'Tool calls must be a list')
        for call in calls:
            require(isinstance(call, dict) and isinstance(call.get('function'), dict),
                    'Malformed function call')
            function = call['function']
            arguments = function.get('arguments')
            if isinstance(arguments, str):
                arguments = load_json(arguments)
            require(isinstance(arguments, dict), 'Tool arguments must be a JSON object')
            encoded(arguments)
            function['arguments'] = arguments
    return result


def checked_file(path, expected):
    raw = Path(path).read_bytes()
    require(sha256(raw) == expected, f'Source/artifact hash mismatch: {path}')
    return raw


class Corpus:
    """Pinned, previously source-recomputed corpus; no new chemical audit."""

    def __init__(self, base):
        self.base = Path(base).resolve()
        build = self.base / 'build-v1'
        self.manifest = load_json(checked_file(build / 'manifest.json', MANIFEST_SHA256))
        checked_file(self.base / 'build_references.py', self.manifest['builder_sha256'])
        for name, digest in self.manifest['source_sha256'].items():
            checked_file(self.base / 'inputs' / name, digest)
        artifacts = {name: checked_file(build / name, digest)
                     for name, digest in self.manifest['artifact_sha256'].items()}
        split = self.manifest['split']
        train = set(split['train'])
        self.heldout = set(split['test'])
        require(len(train) == 150 and len(self.heldout) == 40 and not train & self.heldout,
                'Invalid train/heldout partition')
        self.references = {}
        for line in artifacts['train_references.jsonl'].splitlines():
            row = load_json(line)
            index = row['idx']
            require(type(index) is int and index in train and index not in self.references,
                    'Invalid/duplicate reference target')
            require(row['route_verdict'] == 'VALID', 'Unsupported reference verdict')
            require(row['steps'] and all(isinstance(step['raw'], str) and step['raw']
                                         for step in row['steps']), 'Empty reference')
            self.references[index] = row
        require(len(self.references) == 103, 'Expected 103 covered train targets')
        evaluation = [load_json(line) for line in artifacts['eval_targets.jsonl'].splitlines()]
        require(len(evaluation) == 40 and {row['idx'] for row in evaluation} == self.heldout,
                'Heldout artifact mismatch')

    def reference(self, index, target):
        require(type(index) is int, 'Target index must be an integer')
        require(index not in self.heldout, 'Held-out target forbidden')
        require(index in self.references, 'Unknown or unsupported/uncovered target')
        reference = self.references[index]
        require(reference['target_smiles'] == target, 'Target identity mismatch')
        return reference


def validate_conversation(messages, tools):
    require(isinstance(tools, list) and tools, 'Native tool schemas required')
    names = set()
    for tool in tools:
        require(isinstance(tool, dict) and tool.get('type') == 'function',
                'Unsupported tool schema')
        function = tool.get('function')
        require(isinstance(function, dict), 'Missing function schema')
        name = function.get('name')
        require(isinstance(name, str) and name and name not in names,
                'Invalid/duplicate tool schema name')
        require(isinstance(function.get('parameters'), dict)
                and function['parameters'].get('type') == 'object', 'Invalid parameter schema')
        names.add(name)
    require(messages and messages[-1].get('role') in ('user', 'tool'),
            'Expected complete decision-state history ending with user or tool')
    task_user = next((message for message in messages if message.get('role') == 'user'), None)
    require(task_user is not None and isinstance(task_user.get('content'), str)
            and task_user['content'].strip(), 'First task user must be nonempty text')
    calls, pending = {}, set()
    for message in messages:
        role = message.get('role')
        require(role in ('system', 'user', 'assistant', 'tool'), 'Unsupported native role')
        if role != 'tool':
            require(not pending, 'Tool results missing before next non-tool message')
        if 'tool_calls' in message:
            require(role == 'assistant' and isinstance(message['tool_calls'], list),
                    'Tool calls belong to assistant messages')
            for call in message['tool_calls']:
                identifier = call.get('id')
                require(isinstance(identifier, str) and identifier and identifier not in calls,
                        'Missing/duplicate tool call identity')
                require(call.get('type') == 'function' and call['function'].get('name') in names,
                        'Unknown function or call type')
                calls[identifier] = call['function']['name']
                pending.add(identifier)
        if role == 'tool':
            identifier = message.get('tool_call_id')
            require(identifier in pending, 'Orphan or duplicate tool observation')
            require('name' not in message or message['name'] == calls[identifier],
                    'Tool observation name mismatch')
            require('content' in message, 'Missing tool observation content')
            pending.remove(identifier)
    require(not pending, 'Incomplete tool observations')


def validate_checkpoint(checkpoint):
    require(isinstance(checkpoint, dict) and set(checkpoint) == {'id', 'sha256'},
            'Checkpoint must specify exact id and sha256')
    require(isinstance(checkpoint['id'], str) and checkpoint['id'], 'Missing checkpoint ID')
    digest = checkpoint['sha256']
    require(isinstance(digest, str) and len(digest) == 64
            and all(char in '0123456789abcdef' for char in digest), 'Invalid checkpoint hash')


def convert(source, corpus, *, mode, expected_source_sha256, expected_model,
            expected_checkpoint=None, capture=None):
    require(mode in ('format-probe', 'native-candidate'), 'Unsupported conversion mode')
    require(isinstance(expected_model, str) and expected_model, 'Expected API model required')
    source = Path(source).resolve()
    raw = checked_file(source, expected_source_sha256)
    if mode == 'format-probe':
        packet = load_json(raw)
        require(isinstance(packet, dict) and isinstance(packet.get('requests'), list),
                'Expected old native request packet')
        require(expected_checkpoint is None and capture is None,
                'Old teacher packet cannot claim student checkpoint/capture provenance')
        identity = packet
        entries = packet['requests']
        origin = 'old-teacher-format-only'
        upstream = {'path': packet['source'], 'sha256_claimed': packet['sha256'],
                    'bytes_verified': False}
    else:
        validate_checkpoint(expected_checkpoint)
        require(isinstance(capture, dict) and capture.get('origin') == 'student',
                'Student capture declaration required, but is not proof')
        require(capture.get('checkpoint') == expected_checkpoint, 'Capture checkpoint mismatch')
        identity = capture
        entries = [{'source_line': index, 'record': load_json(line)}
                   for index, line in enumerate(raw.splitlines(), 1)]
        origin = 'declared-student-unverified'
        upstream = None
    require(isinstance(entries, list) and entries, 'Empty request log')
    reference = corpus.reference(identity['case_idx'], identity['target_smiles'])
    privileged = '\n'.join(step['raw'] for step in reference['steps'])
    seen = {name: set() for name in ('source_line', 'request_index', 'request_id', 'llm_call_id')}
    rows = []
    schema_hash = None
    for entry in entries:
        require(isinstance(entry, dict) and isinstance(entry.get('record'), dict),
                'Expected request entry and record objects')
        record = entry['record']
        require(type(record.get('schema_version')) is int and record['schema_version'] == 1
                and record.get('phase') == 'main',
                'Only native schema-v1 main decision requests are supported')
        for field in ('case_idx', 'target_smiles'):
            require(field not in record or (type(record[field]) is type(identity[field])
                    and record[field] == identity[field]), f'Record {field} mismatch')
        require(record.get('model_settings', {}).get('model') == expected_model,
                'Configured model mismatch')
        require('api_model' not in record or record['api_model'] == expected_model,
                'API model mismatch')
        if mode == 'native-candidate':
            require(record.get('checkpoint') == expected_checkpoint, 'Record checkpoint mismatch')
            require(type(record.get('turn')) is int and record['turn'] >= 0,
                    'Native capture turn required')
            for field in ('llm_call_id', 'request_id', 'api_model'):
                require(isinstance(record.get(field), str) and record[field],
                        f'Native capture {field} required')
        else:
            require(record.get('checkpoint') is None, 'Old teacher checkpoint is unknown')
        for field in seen:
            value = entry.get(field) if field == 'source_line' else record.get(field)
            if field in ('source_line', 'request_index'):
                require(type(value) is int and value > 0, f'Invalid {field}')
            if value is not None:
                require(isinstance(value, (str, int)) and value not in seen[field],
                        f'Duplicate/invalid request identity: {field}')
                seen[field].add(value)
        messages = normalize_messages(record['messages'])
        tools = copy.deepcopy(record['tools'])
        validate_conversation(messages, tools)
        current_schema_hash = sha256(encoded(tools))
        require(schema_hash is None or current_schema_hash == schema_hash, 'Tool schema drift')
        schema_hash = current_schema_hash
        student_text = encoded(messages).decode('utf-8')
        require('<privileged_reference>' not in student_text
                and not any(privileged in str(message.get('content', '')) for message in messages),
                'Privileged reference already present in student history')
        require(any(identity['target_smiles'] in str(message.get('content', ''))
                    for message in messages if message['role'] in ('system', 'user')),
                'Target missing from task context')
        provenance = {
            'mode': mode, 'origin': origin, 'on_policy_verified': False,
            'source_path': str(source), 'source_sha256': sha256(raw),
            'source_line': entry['source_line'], 'upstream_log': upstream,
            'record_canonical_sha256': sha256(encoded(record)),
            'original_messages_canonical_sha256': sha256(encoded(record['messages'])),
            'record': {key: copy.deepcopy(value) for key, value in record.items()
                       if key not in ('messages', 'tools')},
            'case_idx': identity['case_idx'], 'target_smiles': identity['target_smiles'],
            'checkpoint': copy.deepcopy(expected_checkpoint),
            'capture_declaration': copy.deepcopy(capture),
            'reference_uid': reference['uid'], 'reference_sha256': sha256(encoded(reference)),
            'reference_sources': copy.deepcopy(reference['sources']),
            'corpus_path': str(corpus.base), 'manifest_sha256': MANIFEST_SHA256,
            'reference_artifact_sha256': corpus.manifest['artifact_sha256']['train_references.jsonl'],
            'tools_sha256': schema_hash,
            'phase': record['phase'], 'turn': record.get('turn'),
            'llm_call_id': record.get('llm_call_id'), 'request_id': record.get('request_id'),
            'api_model': record.get('api_model'),
        }
        rows.append({'prompt': messages, 'privileged_context': privileged, 'tools': tools,
                     'provenance': provenance, 'formal_training_admissible': False})
    receipt = {
        'schema_version': 1, 'mode': mode, 'origin': origin,
        'formal_training_admissible': False, 'on_policy_verified': False,
        'source_sha256': sha256(raw), 'source_bytes': len(raw),
        'manifest_sha256': MANIFEST_SHA256, 'upstream_log': upstream,
        'reason': 'Formatting and byte integrity only; no independent live checkpoint/service evidence.',
        'measurements': [{'source_line': row['provenance']['source_line'],
                          'messages': len(row['prompt']), 'tools': len(row['tools']),
                          'roles': [message['role'] for message in row['prompt']],
                          'prompt_json_bytes': len(encoded(row['prompt'])),
                          'prompt_sha256': sha256(encoded(row['prompt'])),
                          'reference_utf8_bytes': len(row['privileged_context'].encode('utf-8')),
                          'reference_sha256': sha256(row['privileged_context'].encode('utf-8')),
                          'tools_sha256': row['provenance']['tools_sha256']}
                         for row in rows],
    }
    receipt['rows_sha256'] = sha256(b''.join(encoded(row) + b'\n' for row in rows))
    return rows, receipt


def formal_training_rows(rows, receipt):
    """Never treat a local receipt or origin declaration as live-capture authority."""
    raise ValueError('Formal training admission unavailable: independently verified fresh '
                     'student rollout/service/checkpoint evidence is required, not a format receipt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--expected-model', required=True)
    parser.add_argument('--mode', choices=('format-probe', 'native-candidate'), required=True)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite output directory')
    rows, receipt = convert(
        args.source, Corpus(args.corpus), mode=args.mode,
        expected_source_sha256=args.source_sha256, expected_model=args.expected_model,
        capture=load_json(args.capture.read_bytes()) if args.capture else None,
        expected_checkpoint=load_json(args.checkpoint.read_bytes()) if args.checkpoint else None)
    args.output.mkdir()
    (args.output / 'FORMAT_ONLY_NOT_FOR_TRAINING.jsonl').write_bytes(
        b''.join(encoded(row) + b'\n' for row in rows))
    (args.output / 'receipt.json').write_bytes(encoded(receipt) + b'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
