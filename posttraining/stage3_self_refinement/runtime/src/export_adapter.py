"""Export only the terminal, evidence-bound full checkpoint. No tensors loaded here."""
import argparse
from pathlib import Path
import shutil
from contract import require, read_json, sha, write_new, PREDECESSOR_SHA
from evidence_gates import verify_optimizer, ref
ROOT=Path(__file__).resolve().parents[1]

def export(run, checkpoint, owner_validation, out, pins):
    run=Path(run).resolve();ck=Path(checkpoint).resolve();out=Path(out).resolve()
    require(out.is_relative_to(ROOT) and not out.exists(),'new runtime export required')
    evidence=verify_optimizer(owner_validation,pins,stage='full')
    require(evidence['run']==run and evidence['checkpoint']==ck,'export checkpoint/run swapped')
    out.mkdir(parents=True)
    for name in ('adapter_model.safetensors','adapter_config.json'):shutil.copyfile(ck/name,out/name)
    launch=evidence['launch']
    write_new(out/'TRAINED.json',dict(schema='s3.trained.v4',stage='full',trained=True,
        adapter_sha256=sha(out/'adapter_model.safetensors'),adapter_config_sha256=sha(out/'adapter_config.json'),
        predecessor_sha256=PREDECESSOR_SHA,fresh_optimizer=True,manifest_sha256=launch['manifest_sha256'],
        optimizer_steps=evidence['state']['global_step'],terminal_epoch=evidence['state']['epoch'],
        validation=ref(owner_validation),source_pins=ref(pins),terminal_trainer_state=ref(ck/'trainer_state.json')))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--owner-validation',required=True);p.add_argument('--out',required=True)
    p.add_argument('--pins',default=str(ROOT/'SOURCE_PINS.fix-v4-final2.json'));a=p.parse_args()
    export(a.run,a.checkpoint,a.owner_validation,a.out,a.pins)
if __name__=='__main__':main()
