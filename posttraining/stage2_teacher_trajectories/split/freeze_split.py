"""Freeze the stage-2 dev split from the existing Retro*-190 seed-42 train side.

# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_WORKSPACE = os.environ.get("RETRACE_WORKSPACE", "/path/to/workspace")
RETRACE_ARCHIVE = os.environ.get("RETRACE_ARCHIVE", "/path/to/archive")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")

Dev targets are selected by whole Butina clusters (same method as the frozen
150/40 split: Morgan r=2, 2048 bit, Tanimoto 0.6) so analogs never straddle
train/dev. test40 is untouched and remains the evaluation set.
"""
import csv, json, hashlib, random
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

SRC = Path(RETRACE_WORKSPACE + '/retro190-split/data-prep/inputs')
OUT = Path(__file__).resolve().parent
split_bytes = (SRC / 'retro190_split_seed42.json').read_bytes()
split = json.loads(split_bytes)
cases = {int(r['idx']): r for r in csv.DictReader(open(SRC / 'cases.csv'))}
train, test = sorted(split['train']), sorted(split['test'])
assert len(train) == 150 and len(test) == 40 and not set(train) & set(test)

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = [gen.GetFingerprint(Chem.MolFromSmiles(cases[i]['smiles'])) for i in train]
dists = []
for i in range(1, len(fps)):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend(1 - s for s in sims)
clusters = Butina.ClusterData(dists, len(fps), 0.4, isDistData=True)
clusters = [sorted(train[j] for j in c) for c in clusters]

rng = random.Random(20261101)
order = list(range(len(clusters)))
rng.shuffle(order)
TARGET = 20
dev = []
for ci in order:
    if len(dev) >= TARGET:
        break
    if len(dev) + len(clusters[ci]) > TARGET + 2:
        continue
    dev.extend(clusters[ci])
dev = sorted(dev)
train_stage2 = sorted(set(train) - set(dev))

# leakage check: max Tanimoto between dev and remaining train
idx = {t: k for k, t in enumerate(train)}
max_dev_train = max(
    DataStructs.TanimotoSimilarity(fps[idx[d]], fps[idx[t]]) for d in dev for t in train_stage2)

out = {
    'source_split': str(SRC / 'retro190_split_seed42.json'),
    'source_split_sha256': hashlib.sha256(split_bytes).hexdigest(),
    'method': 'Butina clusters on train150 (Morgan r2 2048, Tanimoto 0.6); whole clusters shuffled with seed 20261101, filled to ~20',
    'n_clusters_train150': len(clusters),
    'train_stage2': train_stage2,
    'dev': dev,
    'test40_eval_only': test,
    'n': {'train_stage2': len(train_stage2), 'dev': len(dev), 'test40': len(test)},
    'max_tanimoto_dev_vs_train_stage2': round(max_dev_train, 3),
    'rules': [
        'test40 is evaluation only: no teacher collection, selector calibration, threshold or checkpoint choice.',
        'dev: selector/threshold calibration and checkpoint promotion only; no gradient.',
        'All trajectories, segments and derived samples of one target stay on its side.',
    ],
}
(OUT / 'SPLIT_STAGE2.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({k: v for k, v in out.items() if k not in ('train_stage2', 'test40_eval_only')}, indent=2))
