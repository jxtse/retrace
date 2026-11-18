"""Static (no-launch) check of the OPSD test40 eval path: rebuild the config + argv exactly
as the driver's _eval_launch does, and assert every referenced file exists and the diff vs
the frozen S3 test40 manifest is only the model alias / out dir / workers."""
import hashlib
import json
from pathlib import Path

import yaml


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
PREPARED = Path(RETRACE_SHARE + '/stage3-endtoend/authorized-training/runtime')
EVAL_ALIAS = 'qwen3.8-27b-opsd'
OUT = Path(RETRACE_SHARE + '/online-arms/s3/eval/_drycheck')

for half in ('h0', 'h1'):
    m = json.loads((PREPARED / f'test40-s3-{half}-001' / 'launch_manifest.json').read_text())
    cfg = yaml.safe_load(Path(m['config']['path']).read_text())
    assert cfg['llm']['base_url'].endswith(('8000/v1', '8002/v1')), cfg['llm']['base_url']
    cfg['llm']['model'] = EVAL_ALIAS
    cpath = OUT / f'{half}.yaml'
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(yaml.safe_dump(cfg, sort_keys=False))
    argv = list(m['argv'])
    argv[argv.index('--out') + 1] = str(OUT)
    argv[argv.index('--output-dir') + 1] = str(OUT)
    assert Path(argv[0]).exists(), argv[0]
    assert Path(argv[1]).exists(), argv[1]
    harness = Path(m['cwd'])
    assert (harness / 'dataset/retro-190_dataset/run_benchmark.py').exists()
    env = dict(m['env'])
    print(json.dumps(dict(
        half=half, n_indices=len(m['indices']), indices=m['indices'][:6],
        model=cfg['llm']['model'], temperature=cfg['llm']['temperature'],
        reasoning=cfg['llm']['reasoning_effort'], max_tokens=cfg['llm']['max_tokens'],
        iterations=cfg['agent']['max_iterations'],
        workers=argv[argv.index('--workers') + 1],
        python=argv[0].split('/')[-4:], cwd=str(harness),
        env_keys=sorted(env), tool_url=env.get('RETROPLANNER_HTTP_URL'),
        config_sha_orig=m['config']['sha256'],
        config_sha_new=hashlib.sha256(cpath.read_bytes()).hexdigest(),
        lora_arg=[a for a in m.get('serving_argv', []) if 'lora' in a][:2],
        diff=['llm.model -> ' + EVAL_ALIAS, 'out dir -> ' + str(OUT)],
    )))
print('EVAL_DRYCHECK_OK')
