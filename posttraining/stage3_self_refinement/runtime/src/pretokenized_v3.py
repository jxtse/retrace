"""Production registered native template. Import never changes CUDA or loads weights."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from contract import Release, require, validate_tokens
from swift.template import TEMPLATE_MAPPING, register_template

TEMPLATE = 'retrace_s3_final_selected_v3'
Base = TEMPLATE_MAPPING['qwen3_8'].template_cls


class TextOnlyQwen(Base):
    def _get_model(self):
        raise RuntimeError('text-only rope helper must not construct a model, including meta')

    def _get_get_rope_index(self):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
        cfg = json.loads((Path(self.model_info.model_dir) / 'config.json').read_text())
        holder = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(**cfg['vision_config'])))
        def text_only(*args, **kwargs):
            require(kwargs.get('image_grid_thw') is None and kwargs.get('video_grid_thw') is None, 'text only')
            kinds = kwargs.get('mm_token_type_ids')
            require(kinds is not None and not bool(kinds.any()), 'non-text token types forbidden')
            return Qwen3_5Model.get_rope_index(holder, *args, **kwargs)
        return text_only


class FinalSelectedTemplate(TextOnlyQwen):
    def encode(self, inputs, return_template_inputs=False, return_length=False):
        require(not return_template_inputs, 'template reconstruction unsupported')
        require(set(inputs) == {'uid', 'mode', 'input_ids', 'labels', 'lengths'}, 'unexpected cached columns')
        require(inputs['mode'] == 'FINAL_SELECTED', 'production accepts FINAL_SELECTED only')
        if not hasattr(self, '_release'):
            self._release = Release(os.environ['S3_MANIFEST'], os.environ['S3_MANIFEST_SHA256'])
        p = self._release.payload(inputs['uid'])
        think = [self.tokenizer.convert_tokens_to_ids(x) for x in ('<think>', '</think>')]
        require(think == self._release.manifest['think_token_ids'], 'tokenizer mismatch')
        validate_tokens(p, think, positive=True)
        require(inputs == self._release.record(inputs['uid']), 'cached IDs/labels differ from frozen manifest')
        # Actual reader visits; all SP ranks log separately. This records exposure to the
        # loader, not proof of an optimizer update or successful training.
        if os.environ.get('S3_EXPOSURE_DIR'):
            out = Path(os.environ['S3_EXPOSURE_DIR'])
            with (out / ('reader-rank-' + os.environ.get('RANK', '0') + '.jsonl')).open('a') as f:
                f.write(json.dumps(dict(uid=inputs['uid'], manifest_sha256=self._release.sha256,
                                        full_tokens=len(p['input_ids']), selected_tokens=sum(p['selected_mask']))) + '\n')
        encoded = dict(input_ids=p['input_ids'].copy(), labels=p['labels'].copy())
        if os.environ.get('S3_EVIDENCE_DIR'):
            encoded['_s3_uid'] = inputs['uid']
        if return_length:
            encoded['length'] = len(p['input_ids'])
        return encoded

    def data_collator(self, batch, *, padding_to=None):
        uids = [b.pop('_s3_uid', None) for b in batch]
        result = super().data_collator(batch, padding_to=padding_to)
        if os.environ.get('S3_EVIDENCE_DIR'):
            require(all(type(u) is str for u in uids), 'missing optimizer UID provenance')
            result['_s3_uids'] = uids
        return result


meta = copy.deepcopy(TEMPLATE_MAPPING['qwen3_8'])
meta.template_type = TEMPLATE
meta.template_cls = FinalSelectedTemplate
register_template(meta, exist_ok=True)
