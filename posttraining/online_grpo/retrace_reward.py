"""Reward-function plugin: look up precomputed rewards by trajectory uid.

Reward values themselves are produced outside this build (frozen judge on the collected
routes). In the S1 smoke the table holds clearly-labelled FIXTURE values.
"""
import json
import os
from pathlib import Path

from swift.rewards.orm import ORM
from swift.rewards import orms


class RetraceRewardLookup(ORM):

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        path = os.environ.get('RETRACE_REWARDS')
        if not path:
            raise RuntimeError('RETRACE_REWARDS must point at the reward lookup table')
        self.table = json.loads(Path(path).read_text())
        self.is_fixture = bool(self.table.get('__fixture__'))
        self.values = {k: v for k, v in self.table.items() if not k.startswith('__')}
        self.hits = 0
        self.misses = []

    def __call__(self, completions=None, **kwargs):
        keys = kwargs.get('uid') or kwargs.get('request_id')
        if keys is None:
            raise RuntimeError('reward lookup needs a uid/request_id column')
        out = []
        for k in keys:
            if k not in self.values:
                self.misses.append(k)
                out.append(0.0)
            else:
                self.hits += 1
                out.append(float(self.values[k]))
        return out


orms['retrace_reward_lookup'] = RetraceRewardLookup
