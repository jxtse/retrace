"""GRPO replay seam: the ONLY override is the obtain-completions stage (`_rollout_samples`).

Completions (token ids + loss mask) come from the collected native-harness trajectories in
the dataset rows, so we skip vLLM generation entirely. Rewards arrive through the
`retrace_reward_lookup` ORM plugin; advantages, KL and loss stay stock ms-swift 4.6.0.
"""
import json

from swift.rlhf_trainers import GRPOTrainer


class ReplayGRPOTrainer(GRPOTrainer):

    def _rollout_samples(self, inputs):
        samples = self.to_samples(inputs)  # rows -> GRPOSample (messages + uid already present)
        for s in samples:
            # Provenance only: no generation happened, so mark the completion as replay.
            s.rollout_infos = dict(s.rollout_infos or {}, replay=True)
            if s.finish_reason is None:
                s.finish_reason = 'replay'
        return samples
