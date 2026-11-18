#!/usr/bin/env python3
"""Start one OPSD vLLM replica (frozen test40 serving argv, TP2, port 8000/8002).

usage: opsd_serve.py --half h0 --port 8000 --cuda 2,3 --dir <serving dir> \
                     --base-alias qwen3.8-27b-opsd-base [--lora-alias X --adapter DIR]
The only deltas vs the frozen base/S3/teacher-SFT serving argv are the model aliases and
the LoRA module path; everything else (dtype, max-model-len 131072, gpu-memory 0.90,
max-num-seqs 8, tool/reasoning parsers, TP2, --enable-lora) is unchanged.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
MODEL = RETRACE_SHARE + '/models/Qwen/Qwen3.8-27B'
SERVING_PY = RETRACE_SHARE + '/stage3-endtoend/runtime/envs-attempt1/serving/bin/python'
VLLM_BIN = RETRACE_SHARE + '/retrace-continual/venv-vllm/bin/vllm'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ticks(pid):
    return int(pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--half', required=True)
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--cuda', required=True)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--base-alias', required=True)
    ap.add_argument('--lora-alias', default=None)
    ap.add_argument('--adapter', default=None)
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    argv = ['timeout', '--signal=TERM', '--kill-after=60s', '86400', SERVING_PY, VLLM_BIN, 'serve', MODEL,
            '--served-model-name', a.base_alias, '--host', '127.0.0.1', '--port', str(a.port),
            '--dtype', 'bfloat16', '--max-model-len', '131072', '--gpu-memory-utilization', '0.90',
            '--max-num-seqs', '8', '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_xml',
            '--reasoning-parser', 'qwen3', '--language-model-only', '--generation-config', 'auto',
            '--tensor-parallel-size', '2']
    if a.adapter:
        assert a.lora_alias, 'lora alias required with an adapter'
        argv += ['--enable-lora', '--max-lora-rank', '32', '--max-loras', '1',
                 '--lora-modules', f'{a.lora_alias}={a.adapter}']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.cuda, VLLM_USE_V1='1')
    for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        env.pop(k, None)
    log = open(d / 'server.log', 'a')
    pr = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=log,
                          stderr=subprocess.STDOUT, start_new_session=True)
    owner = dict(
        identity=dict(pid=pr.pid, start_ticks=ticks(pr.pid), pgrp=pr.pid,
                      boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()),
        argv=argv, cuda=a.cuda, half=a.half, port=a.port,
        base_alias=a.base_alias, lora_alias=a.lora_alias, adapter=a.adapter,
        adapter_sha256=sha(pathlib.Path(a.adapter) / 'adapter_model.safetensors') if a.adapter else None,
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (d / 'OWNER.json').write_text(json.dumps(owner, indent=1))
    print(json.dumps({'SERVER_STARTED': a.half, 'pid': pr.pid, 'port': a.port, 'cuda': a.cuda}))


if __name__ == '__main__':
    main()
