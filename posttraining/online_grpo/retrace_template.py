"""Registered replay template for the GRPO/OPSD arms (our code only; framework untouched).

Why this exists: the native dataset loader rewrites rows into ms-swift's internal message
convention (an assistant turn's OpenAI ``tool_calls`` become separate ``tool_call`` messages).
The framework's own ``Template.encode`` knows how to re-merge and render that convention;
the plain chat template / our raw renderer do not. So this template delegates *rendering*
to the stock template and only replaces the loss mask (mask = every model-generated token,
0 on user/tool/template tokens), then validates the result against a manifest of
sha256(input_ids) and loss-token counts. Any dataset round-trip that changes the tokens
fails closed instead of silently training on drifted ids.

Loss mask is computed without re-rendering: every assistant generation starts at the
``<|im_start|>assistant\n`` (+ ``<think>\n``) marker and ends at the next ``<|im_end|>``.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swift.template import TEMPLATE_MAPPING, register_template  # noqa: E402

TEMPLATE = 'retrace_opsd_replay_v1'
Base = TEMPLATE_MAPPING['qwen3_8'].template_cls
_MANIFEST = None


def manifest():
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = json.loads(Path(os.environ['RETRACE_MANIFEST']).read_text())
    return _MANIFEST


def sha(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


def generation_spans(ids, tok, include_eos=False):
    """Locate every model-generated span in an already rendered id sequence.

    A generation starts right after the ``<|im_start|>assistant\n`` header (plus the
    template's ``<think>\n`` thinking prefix, which is prompt context, not sampled) and
    ends at the next ``<|im_end|>``.
    """
    im_start = tok.convert_tokens_to_ids('<|im_start|>')
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    spans = []
    n = len(ids)
    i = 0
    while i < n:
        if ids[i] != im_start:
            i += 1
            continue
        head = tok.decode(ids[i:i + 12])
        if not head.startswith('<|im_start|>assistant'):
            i += 1
            continue
        hdr_chars = head.index('\n') + 1
        if head[hdr_chars:].startswith('<think>\n'):
            hdr_chars += len('<think>\n')
        m = 1
        while m <= 12 and len(tok.decode(ids[i:i + m])) < hdr_chars:
            m += 1
        j = i + m
        while j < n and ids[j] != im_end:
            j += 1
        start, end = i + m, (j + 1 if include_eos else j)
        if end > start:
            spans.append((start, end))
        i = max(j, i + 1)
    return spans


class ReplayTemplate(Base):

    def encode(self, inputs, return_template_inputs=False, return_length=False):
        if return_template_inputs:
            raise RuntimeError('template reconstruction is not supported for replay rows')
        ctk = inputs.get('chat_template_kwargs') or {}
        uid = ctk.get('_retrace_uid')
        if uid is None:
            raise RuntimeError('row is missing chat_template_kwargs._retrace_uid')
        entry = manifest()[uid]
        # TITO: rows carry exact per-turn assistant ids; inject them as content so the
        # stock encoder emits those tokens verbatim instead of re-rendering tool calls.
        rt = inputs.get('response_token_ids')
        if rt:
            masks = inputs.get('response_loss_mask') or [[1] * len(x) for x in rt]
            msgs = [dict(m) for m in (json.loads(inputs['messages'])
                                      if isinstance(inputs['messages'], str) else inputs['messages'])]
            turn_idx = [i for i, m in enumerate(msgs) if m.get('role') == 'assistant']
            for k, i in enumerate(sorted(turn_idx, reverse=True)):
                if k < len(rt):
                    msgs[i]['content'] = {'token_ids': rt[-1 - k], 'loss_scale': masks[-1 - k]}
            inputs = dict(inputs, messages=msgs)
        rendered = super().encode(inputs, return_length=True)
        ids = list(rendered['input_ids'])
        tok = self.tokenizer
        spans = generation_spans(ids, tok)
        labels = [-100] * len(ids)
        for s, e in spans:
            labels[s:e] = ids[s:e]
        n_loss = sum(1 for x in labels if x != -100)
        # teacher view = privileged block appended to the last user message
        block = (entry.get('teacher') or {}).get('block')
        key = 'teacher' if (block and self._last_user_ends_with(inputs, block)) else 'student'
        exp = entry[key]
        if len(ids) != exp['n_tokens'] or sha(ids) != exp['ids_sha256']:
            import logging
            msg = (f'{key} encoding drift uid={uid}: got {len(ids)} tokens, '
                   f'expected {exp["n_tokens"]} (delta {len(ids) - exp["n_tokens"]})')
            if os.environ.get('RETRACE_ASSERT', 'warn') == 'raise':
                raise RuntimeError(msg)
            logging.getLogger('swift').warning('RETRACE_PARITY ' + msg)
            n_loss = sum(1 for x in labels if x != -100)
            out = {'input_ids': ids, 'labels': labels}
            if return_length:
                out['length'] = len(ids)
            return out
        if False:
            raise RuntimeError(
                f'{key} encoding drift uid={uid}: got {len(ids)} tokens {sha(ids)[:12]}, '
                f'expected {exp["n_tokens"]} {exp["ids_sha256"][:12]}'
                ' (dataset round-trip changed the tokens)')
        if os.environ.get('RETRACE_ASSERT_LOSS', '1') == '1' and n_loss != exp['n_loss']:
            raise RuntimeError(f'{key} loss-mask drift uid={uid}: got {n_loss}, expected {exp["n_loss"]}')
        out = {'input_ids': ids, 'labels': labels}
        if return_length:
            out['length'] = len(ids)
        return out

    @staticmethod
    def _last_user_ends_with(inputs, block):
        messages = inputs.get('messages') or []
        if isinstance(messages, str):
            messages = json.loads(messages)
        for m in reversed(messages):
            if m.get('role') == 'user':
                c = m.get('content')
                return isinstance(c, str) and c.endswith(block)
        return False


meta = TEMPLATE_MAPPING['qwen3_8']
meta.template_type = TEMPLATE
meta.template_cls = ReplayTemplate
register_template(meta, exist_ok=True)
