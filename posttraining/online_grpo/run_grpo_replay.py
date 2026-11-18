"""S1 runner: one GRPO replay step with looked-up (fixture) rewards.

Only seam: TrainerFactory maps 'grpo' to our subclass, which skips generation in
`_rollout_samples`. Everything else (rewards -> advantages -> loss) is stock ms-swift.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import retrace_template  # noqa: F401  registers `retrace_opsd_replay_v1`
import retrace_reward  # noqa: F401  registers `retrace_reward_lookup`
from swift.pipelines.train.rlhf import SwiftRLHF  # noqa: E402
from swift.trainers.trainer_factory import TrainerFactory  # noqa: E402

TrainerFactory.TRAINER_MAPPING['grpo'] = 'replay_grpo_trainer.ReplayGRPOTrainer'

if __name__ == '__main__':
    SwiftRLHF(sys.argv[1:]).main()
