#!/usr/bin/env python3
"""Stop OPSD vLLM replicas recorded in OWNER.json files (only processes we started).

usage: opsd_stop_serve.py <serving dir with h0/, h1/> ...
Verifies the recorded pid is still the same process (pid + start_ticks + boot_id) before
signalling; waits for the port to close and for GPU memory to drop.
"""
import json
import os
import pathlib
import signal
import socket
import sys
import time

BOOT = pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def ticks(pid):
    return int(pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])


def state(pid):
    try:
        f = pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return f[0], int(f[19])
    except (FileNotFoundError, ProcessLookupError):
        return None, None


def port_open(port):
    s = socket.socket()
    s.settimeout(2)
    try:
        return s.connect_ex(('127.0.0.1', port)) == 0
    finally:
        s.close()


def gpu_used():
    import subprocess
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    return {int(l.split(',')[0]): int(l.split(',')[1]) for l in out.strip().splitlines()}


def main():
    dirs = [pathlib.Path(p) for p in sys.argv[1:]]
    stopped = []
    for d in dirs:
        owner = d / 'OWNER.json'
        if not owner.exists():
            continue
        o = json.loads(owner.read_text())
        ident = o['identity']
        st, tk = state(ident['pid'])
        if st is not None and tk == ident['start_ticks'] and ident.get('boot_id', BOOT) == BOOT:
            try:
                os.killpg(ident['pgrp'], signal.SIGTERM)
                stopped.append(dict(dir=str(d), pid=ident['pid'], signal='SIGTERM'))
            except ProcessLookupError:
                stopped.append(dict(dir=str(d), pid=ident['pid'], signal='already-gone'))
        else:
            stopped.append(dict(dir=str(d), pid=ident['pid'], signal='identity-mismatch-skipped'))
        deadline = time.time() + 240
        while time.time() < deadline:
            if not port_open(o['port']):
                break
            time.sleep(3)
    deadline = time.time() + 300
    while time.time() < deadline:
        u = gpu_used()
        if all(v < 1000 for v in u.values()):
            break
        time.sleep(5)
    all_open = {o['port']: port_open(o['port']) for p in dirs
                for o in [json.loads((p / 'OWNER.json').read_text())] if (p / 'OWNER.json').exists()}
    print(json.dumps(dict(stopped=stopped, gpu_used=gpu_used(), ports_open=all_open)))


if __name__ == '__main__':
    main()
