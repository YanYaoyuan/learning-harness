#!/usr/bin/env python3
"""Read saved checkpoints in a separate process; no Trainer or weight updates."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
from learning_harness.common import file_sha,read_jsonl


def main():
    p=argparse.ArgumentParser();p.add_argument('--trajectory',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    import torch
    from transformers import AutoTokenizer
    from learning_harness.adaptation import load_composition
    from train_lora import encode_example
    meta=json.loads((a.trajectory/'trajectory.json').read_text());assert meta['complete']
    tokenizer=AutoTokenizer.from_pretrained(meta['base_model'],local_files_only=True,padding_side='left')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    source=read_jsonl(a.data);examples=[]
    for row in source:
        assert row['partition']=='dev' and row['adaptation_role']=='b_diagnostic'
        # Tokenization/masking reuse only; no optimizer exists in this process.
        encoded=encode_example(tokenizer,dict(row,partition='train'),meta['config']['max_sequence_length'])
        if encoded is None:raise ValueError('Frozen diagnostic eligibility changed')
        examples.append((row['task_id'],encoded))
    path=a.trajectory/'diagnostic-nll.json';result=json.loads(path.read_text()) if path.exists() else {}
    for step in meta['config']['checkpoints']:
        checkpoint=a.trajectory/f'step-{step:04d}';before=file_sha(checkpoint/'B/adapter_model.safetensors')
        if str(step) in result:
            assert result[str(step)]['data_sha256']==file_sha(a.data)
            assert result[str(step)]['checkpoint_sha256']==before
            assert result[str(step)]['separate_process'];continue
        model,revision=load_composition(checkpoint,meta['base_model'],meta['base_revision']);model.eval()
        grouped=defaultdict(list)
        with torch.inference_mode():
            for offset in range(0,len(examples),8):
                batch=examples[offset:offset+8];length=max(len(r['input_ids']) for _,r in batch)
                ids=torch.tensor([[tokenizer.pad_token_id]*(length-len(r['input_ids']))+r['input_ids'] for _,r in batch],device='cuda:0')
                mask=torch.tensor([[0]*(length-len(r['input_ids']))+r['attention_mask'] for _,r in batch],device='cuda:0')
                targets=[[v for v in r['labels'] if v!=-100] for _,r in batch]
                output=model(input_ids=ids,attention_mask=mask,logits_to_keep=max(map(len,targets))+1).logits
                for (task,_),logits,target in zip(batch,output,targets):
                    value=torch.nn.functional.cross_entropy(logits[-len(target)-1:-1].float(),torch.tensor(target,device='cuda:0'),reduction='mean')
                    grouped[task].append(float(value))
        per_task={t:sum(v)/len(v) for t,v in grouped.items()}
        assert file_sha(checkpoint/'B/adapter_model.safetensors')==before
        result[str(step)]=dict(optimizer_step=step,examples=len(examples),tasks=len(per_task),
            task_mean_nll=sum(per_task.values())/len(per_task),per_task_nll=per_task,
            data_sha256=file_sha(a.data),checkpoint_sha256=before,composition_revision=revision,
            used_for_updates=False,used_for_selection=False,separate_process=True,checkpoint_unchanged=True)
        tmp=path.with_suffix('.json.part');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
        print(json.dumps(dict(step=step,task_mean_nll=result[str(step)]['task_mean_nll'])),flush=True)
        del model,output,logits,ids,mask,value;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
