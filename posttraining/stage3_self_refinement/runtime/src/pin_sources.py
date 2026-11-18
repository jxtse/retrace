"""Run once after staging; pin actual installed sources and our code, no weights loaded."""
from pathlib import Path
from contract import sha, write_new

# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
ROOT=Path(__file__).resolve().parents[1]
BASE=Path(RETRACE_SHARE + '')
SITE=BASE/'swift-env/venv-swift/lib/python3.12/site-packages'
paths=list((ROOT/'src').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+list((ROOT/'tests/fixtures').glob('*.json'))
paths += [SITE/p for p in ['swift/template/base.py','swift/template/templates/qwen.py',
    'swift/sequence_parallel/sequence_parallel.py','swift/sequence_parallel/ulysses.py',
    'swift/sequence_parallel/utils.py','swift/pipelines/train/sft.py','swift/pipelines/utils.py',
    'swift/dataset/utils.py','swift/trainers/seq2seq_trainer.py','swift/trainers/mixin.py','swift/pipelines/train/tuner.py','swift/utils/io_utils.py',
    'transformers/trainer.py','swift/callbacks/mapping.py','swift/config/zero3.json','transformers/models/qwen3_5/modeling_qwen3_5.py']]
paths += [BASE/p for p in ['stage2-train/run_a1v2.sh',
    'eval-stage2/launch_split.py','eval-stage2/prepare_isolated.py',
    'retrace-continual/rollouts/eval_round1_test40/base/eval_config.yaml',
    'retrace-continual/rollouts/eval_round1_test40/base/detached_runner.py',
    'retrace-continual/rollouts/eval_round1_test40/base/launch_manifest.json',
    'models/Qwen/Qwen3.8-27B/config.json','models/Qwen/Qwen3.8-27B/tokenizer_config.json',
    'models/Qwen/Qwen3.8-27B/tokenizer.json','models/Qwen/Qwen3.8-27B/chat_template.jinja',
    'models/Qwen/Qwen3.8-27B/generation_config.json',
    'stage3-data-prep/v1/control/INDEX.jsonl',
    'stage3-data-prep/v2/pretokenized.py',
    'retrace-continual/venv-vllm/bin/vllm']]
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--out',default=str(ROOT/'SOURCE_PINS.json')); a=p.parse_args()
    write_new(a.out,{'files':[{'path':str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f),'sha256':sha(f)} for f in paths]})
