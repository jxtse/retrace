#!/usr/bin/env python3
"""Compare the exported OPSD LoRA adapter key set with the teacher-SFT adapter.

usage: opsd_keys.py <our adapter_model.safetensors> <teacher adapter_model.safetensors>
Prints one JSON line: n_ours, n_teacher, equal, only_ours, only_teacher, keys_sha256.
"""
import hashlib
import json
import sys

from safetensors import safe_open


def keys(p):
    with safe_open(p, framework='pt') as f:
        return sorted(f.keys())


def main():
    ours, teach = keys(sys.argv[1]), keys(sys.argv[2])
    only_o = [k for k in ours if k not in teach]
    only_t = [k for k in teach if k not in ours]
    print(json.dumps(dict(
        n_ours=len(ours), n_teacher=len(teach), equal=(ours == teach),
        only_ours=only_o[:5], only_teacher=only_t[:5],
        n_only_ours=len(only_o), n_only_teacher=len(only_t),
        keys_sha256=hashlib.sha256(json.dumps(ours).encode()).hexdigest())))


if __name__ == '__main__':
    main()
