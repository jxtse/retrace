"""Plain torch Dataset that hands the trainer the recorded messages untouched.

Do not use `datasets.Dataset.from_list` for tool-call conversations: Arrow
unifies the struct schema of `function.arguments` across all tool calls, adding
null keys that the Qwen chat template renders as `<parameter=x>null</parameter>`.
SDFTTrainer only special-cases `datasets.Dataset` for column removal and rejects
IterableDataset; a map-style torch Dataset goes through its identity collator.
"""
import torch


class RawPromptDataset(torch.utils.data.Dataset):
    def __init__(self, rows, repeat=1):
        self._rows = [{'prompt': r['prompt'], 'privileged_context': r['privileged_context']} for r in rows] * repeat

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return self._rows[i]
