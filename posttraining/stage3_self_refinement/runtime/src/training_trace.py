"""Owner-run tracing plugin. Import registers hooks; does not build a model or optimizer."""
import inspect
import json
import os
from pathlib import Path
from transformers import Trainer, TrainerCallback
from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
from swift.callbacks import callbacks_map
from contract import require, sha, write_new
from evidence_gates import ref


def location():
    out=Path(os.environ['S3_EVIDENCE_DIR']).resolve()
    require(out.is_relative_to(Path(__file__).resolve().parents[1]),'trace outside runtime')
    return out,int(os.environ.get('RANK','0')),sha(out.parent/'launch.json')


def append_event(name,data):
    out,rank,launch=location()
    with (out/f'{name}-rank-{rank}.jsonl').open('a') as f:
        f.write(json.dumps(dict(data,rank=rank,launch_sha256=launch),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())

_original_schedule=Trainer.set_initial_training_values
_original_step=Seq2SeqTrainer.training_step


def schedule(self,args,dataloader):
    result=_original_schedule(self,args,dataloader)
    out,rank,launch=location()
    keys=('num_train_epochs','num_update_steps_per_epoch','num_examples','num_train_samples','total_train_batch_size','steps_in_epoch','max_steps')
    record=dict(zip(keys,result),origin='actual_trainer_runtime',launch_sha256=launch,epoch_based=args.max_steps<0,
                trainer_source=ref(inspect.getfile(_original_schedule)),dataloader_length=len(dataloader),
                sequence_parallel_size=self.get_sp_size(),gradient_accumulation_steps=args.gradient_accumulation_steps)
    write_new(out/f'trainer-runtime-rank-{rank}.json',record)
    return result


def training_step(self,model,inputs,*args,**kwargs):
    uids=inputs.pop('_s3_uids',None);require(isinstance(uids,list) and uids,'missing collator UID provenance')
    result=_original_step(self,model,inputs,*args,**kwargs)
    if not hasattr(self,'_s3_pending_batches'):self._s3_pending_batches=[]
    self._s3_pending_batches.append(uids)
    return result


class TraceCallback(TrainerCallback):
    def __init__(self,args,trainer):self.trainer=trainer
    def on_step_end(self,args,state,control,**kwargs):
        batches=getattr(self.trainer,'_s3_pending_batches',[])
        require(batches,'optimizer step has no completed microbatches')
        append_event('optimizer-exposure',dict(origin='actual_training_step_commit',step=state.global_step,
            optimizer_step_skipped=bool(self.trainer.accelerator.optimizer_step_was_skipped),
            uids=[u for batch in batches for u in batch],microbatches=len(batches)))
        self.trainer._s3_pending_batches=[]
    def on_log(self,args,state,control,logs=None,**kwargs):
        append_event('trainer-log',dict(step=state.global_step,logs=logs or {}))

Trainer.set_initial_training_values=schedule
Seq2SeqTrainer.training_step=training_step
callbacks_map['s3_trace_v4']=TraceCallback
