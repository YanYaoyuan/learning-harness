#!/usr/bin/env python3
"""One continuous B trajectory, with immutable intermediate compositions."""
import argparse
import json
from pathlib import Path

from learning_harness.common import file_sha,read_jsonl


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='data/models/Qwen3-4B-Instruct-2507')
    p.add_argument('--revision',default='cdbee75f17c01a7cc42f958dc650907174af0554')
    p.add_argument('--a-adapter')
    p.add_argument('--data',default='runs/learning-speed-v1/b_schedule.jsonl')
    p.add_argument('--config',default='runs/learning-speed-v1/adaptation-config.json')
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    import torch
    from transformers import AutoTokenizer,Trainer,TrainingArguments,TrainerCallback
    from learning_harness.adaptation import build,freeze_a_dropout,parameter_hash,save_composition
    from train_lora import encode_example
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty trajectory directory')
    out.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(args.config).read_text());source=read_jsonl(args.data)
    if len(source)!=config['max_steps']*config['global_batch_size']:raise ValueError('Wrong shared sample budget')
    if [r['shared_index'] for r in source]!=list(range(len(source))):raise ValueError('Shared sequence order changed')
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision,padding_side='left',trust_remote_code=False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    examples=[]
    for i,row in enumerate(source):
        e=encode_example(tokenizer,row,config['max_sequence_length'])
        if e is None:raise ValueError('Eligibility must be fixed before comparing initializations')
        examples.append(dict(e,shared_index=i))
    check_strings=[tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=False) for row in source[:2]]
    check=tokenizer(check_strings,add_special_tokens=False,padding=True,return_tensors='pt')
    check={k:v.to('cuda:0') for k,v in check.items()}
    model,initial=build(args.model,args.revision,config,args.a_adapter,seed=args.seed,training=True,check_inputs=check)
    model.config.use_cache=False;model.enable_input_require_grads()
    a_before=parameter_hash(model,'A')
    indices=[];counts=dict(input_tokens=0,target_tokens=0)
    # Identical teacher-forced B material, evaluated before any B update.
    nll_sum=0.;nll_tokens=0;nll_examples=[]
    with torch.inference_mode():
        for offset in range(0,len(examples),8):
            batch=examples[offset:offset+8];length=max(len(r['input_ids']) for r in batch)
            ids=torch.tensor([[tokenizer.pad_token_id]*(length-len(r['input_ids']))+r['input_ids'] for r in batch],device='cuda:0')
            mask=torch.tensor([[0]*(length-len(r['input_ids']))+r['attention_mask'] for r in batch],device='cuda:0')
            targets=[[v for v in r['labels'] if v!=-100] for r in batch]
            logits=model(input_ids=ids,attention_mask=mask,logits_to_keep=max(map(len,targets))+1).logits
            for row,target in zip(logits,targets):
                value=torch.nn.functional.cross_entropy(row[-len(target)-1:-1].float(),torch.tensor(target,device='cuda:0'),reduction='sum')
                nll_sum+=float(value);nll_tokens+=len(target);nll_examples.append(float(value)/len(target))
    initial_nll=dict(shared_examples=len(examples),target_tokens=nll_tokens,token_mean=nll_sum/nll_tokens,
                     example_mean=sum(nll_examples)/len(nll_examples),query_used=False)

    def save(step):
        if parameter_hash(model,'A')!=a_before:raise ValueError('Frozen A weights changed')
        return save_composition(model,out/f'step-{step:04d}',args.model,args.revision,config,args.a_adapter,step,
            dict(training_seed=args.seed,data_sha256=file_sha(args.data),initialization=initial))

    save(0)

    def collate(batch):
        maximum=max(len(r['input_ids']) for r in batch)
        result={key:torch.tensor([r[key]+[pad]*(maximum-len(r[key])) for r in batch]) for key,pad in (
            ('input_ids',tokenizer.pad_token_id),('attention_mask',0),('labels',-100))}
        result['shared_index']=torch.tensor([r['shared_index'] for r in batch])
        return result

    class SharedTrainer(Trainer):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.model_accepts_loss_kwargs=False

        def _get_train_sampler(self,train_dataset=None):
            return torch.utils.data.SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            indices.extend(inputs.pop('shared_index').detach().cpu().tolist())
            freeze_a_dropout(model)
            counts['input_tokens']+=int(inputs['attention_mask'].sum())
            counts['target_tokens']+=int((inputs['labels']!=-100).sum())
            return super().compute_loss(model,inputs,return_outputs,num_items_in_batch=num_items_in_batch)

    class SaveTrajectory(TrainerCallback):
        def on_step_end(self,args,state,control,**kwargs):
            if state.global_step in config['checkpoints']:save(state.global_step)
            return control

    arguments=TrainingArguments(output_dir=str(out),max_steps=config['max_steps'],per_device_train_batch_size=1,
        gradient_accumulation_steps=config['global_batch_size'],learning_rate=config['learning_rate'],
        warmup_ratio=config['warmup_ratio'],weight_decay=config['weight_decay'],bf16=True,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        seed=args.seed,data_seed=config['data_seed'],logging_steps=8,save_strategy='no',report_to=[],
        remove_unused_columns=False)
    trainer=SharedTrainer(model=model,args=arguments,train_dataset=examples,data_collator=collate,callbacks=[SaveTrajectory()])
    trainer.train()
    assert indices==list(range(len(source)))
    assert parameter_hash(model,'A')==a_before
    assert parameter_hash(model,'B')!=initial['b_initial_parameter_sha256']
    metadata=dict(base_model=args.model,base_revision=args.revision,a_adapter=args.a_adapter,b_seed=args.seed,
        config=config,initialization=initial,processed_examples=len(indices),data_sha256=file_sha(args.data),
        actual_order_sha256=__import__('hashlib').sha256(json.dumps(indices).encode()).hexdigest(),
        a_frozen_verified=True,b_updated_verified=True,frozen_a_dropout_disabled=True,token_accounting=counts,
        initial_b_material_nll=initial_nll,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),complete=True)
    (out/'trajectory.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
