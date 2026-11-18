"""No-claim plan fixture: substitute the absent scientific release in memory only.
Never emit an admitted manifest or invoke subprocesses. Production gates stay unchanged.
"""
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import train_s3
from contract import write_new, PREDECESSOR_SHA, CONFIG_SHA
ROOT=Path(__file__).resolve().parents[1]
class PreviewRelease:
    path=Path('/OWNER_SUPPLIED/FINAL_MANIFEST.json')
    rows={'PREVIEW_NOT_TRAINING':{}}
    def payload(self,uid):return {'input_ids':[10,11],'selected_mask':[0,1]}

def main():
    def fake_sha(path):
        if str(path).endswith('adapter_model.safetensors'):return PREDECESSOR_SHA
        if str(path).endswith('adapter_config.json'):return CONFIG_SHA
        return 'PREVIEW_NOT_A_HASH'
    def fake_read(path):
        if str(path).endswith('adapter_config.json'):return {'r':8,'lora_alpha':16}
        return {'passed':True,'actual_manifest_sha256':'OWNER_SUPPLIED_SHA256','actual_final_rows':1}
    with patch.object(train_s3,'Release',return_value=PreviewRelease()),patch.object(train_s3,'sha',side_effect=fake_sha),patch.object(train_s3,'read_json',side_effect=fake_read),patch.object(train_s3,'verify_pins',return_value='OWNER_SUPPLIED_PINS'),patch.object(train_s3,'verify_native',return_value={}):
        _,plan=train_s3.plan('absent','OWNER_SUPPLIED_SHA256','full',ROOT/'FUTURE_FULL_RUN','absent','absent')
        _,smoke=train_s3.plan('absent','OWNER_SUPPLIED_SHA256','smoke',ROOT/'FUTURE_SMOKE_RUN','absent','absent')
    argv=plan['argv']
    for flag,value in {'--sequence_parallel_size':'4','--deepspeed':'zero3','--learning_rate':'2e-5',
        '--padding_free':'true','--gradient_accumulation_steps':'4','--max_length':'131072','--optim':'adamw_torch',
        '--use_liger_kernel':'false','--truncation_strategy':'raise','--num_train_epochs':'1'}.items():
        assert argv[argv.index(flag)+1]==value,(flag,value)
    assert '--resume_from_checkpoint' not in argv and '--max_steps' not in argv
    assert smoke['argv'][smoke['argv'].index('--max_steps')+1]=='2'
    write_new(sys.argv[1],dict(artifact_type='NONEXECUTABLE_ENGINEERING_RECIPE_PREVIEW',training_ready=False,
        model_calls=0,optimizer_steps=0,warning='In-memory test doubles only; no scientific release or data acceptance',full=plan,smoke=smoke))
if __name__=='__main__':main()
