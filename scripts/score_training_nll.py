#!/usr/bin/env python3
"""Read-only base-model action NLL for the two existing training pools."""
import json
import os
from pathlib import Path
import signal
import time

from learning_harness.bounded_queue import save
from learning_harness.common import file_sha, read_jsonl

ROOT=Path('runs/variance-diagnostic-v1/stage3/base-nll')


def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import run_api_harness as engine
    from train_lora import encode_example
    def deadline(signum,frame):
        raise TimeoutError('Read-only NLL scoring reached 900-second preparation cap')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(900)
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'result.json').exists():
        raise ValueError('Scoring result exists; no duplicate GPU job')
    started=time.monotonic()
    record=dict(status='running',gpu=0,cap_gpu_hours=.25,teacher_calls=0,optimizer_updates=0,
        base_model=engine.MODEL,base_revision=engine.REV,script_sha256=file_sha(__file__),sources={})
    save(ROOT/'execution.json',record)
    tokenizer=AutoTokenizer.from_pretrained(engine.MODEL,revision=engine.REV,padding_side='left',local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(engine.MODEL,revision=engine.REV,torch_dtype=torch.bfloat16,
        local_files_only=True).to('cuda:0')
    model.eval()
    token_check=False
    try:
        for source in ('api','generic'):
            data=Path(f'runs/transfer-factorial-v1/corpora/full-{source}.jsonl')
            rows=read_jsonl(data)
            encoded=[encode_example(tokenizer,r,2048) for r in rows]
            if any(r is None for r in encoded):
                raise ValueError('Training eligibility changed')
            scores=[]
            with torch.inference_mode():
                for offset in range(0,len(rows),8):
                    batch=encoded[offset:offset+8]
                    length=max(len(r['input_ids']) for r in batch)
                    ids=torch.tensor([[tokenizer.pad_token_id]*(length-len(r['input_ids']))+r['input_ids'] for r in batch],device='cuda:0')
                    mask=torch.tensor([[0]*(length-len(r['input_ids']))+r['attention_mask'] for r in batch],device='cuda:0')
                    targets=[[x for x in r['labels'] if x!=-100] for r in batch]
                    logits=model(input_ids=ids,attention_mask=mask,logits_to_keep=max(map(len,targets))+1).logits
                    losses=[]
                    for i,target in enumerate(targets):
                        loss=torch.nn.functional.cross_entropy(logits[i,-len(target)-1:-1].float(),
                            torch.tensor(target,device='cuda:0'),reduction='mean')
                        losses.append(float(loss))
                    if not token_check:
                        # Independent per-example forward checks the batched/right-aligned target calculation.
                        one=batch[0]
                        one_ids=torch.tensor([one['input_ids']],device='cuda:0')
                        labels=torch.tensor([one['labels']],device='cuda:0')
                        direct=float(model(input_ids=one_ids,labels=labels).loss)
                        if abs(direct-losses[0])>.03:
                            raise ValueError(f'NLL validation mismatch: {direct} versus {losses[0]}')
                        record['nll_validation']=dict(direct_loss=direct,batched_loss=losses[0],tolerance=.03,passed=True)
                        token_check=True
                    for i,loss in enumerate(losses):
                        index=offset+i
                        scores.append(dict(index=index,task_id=rows[index]['task_id'],mean_target_nll=loss,
                                           target_tokens=len(targets[i]),input_tokens=len(batch[i]['input_ids'])))
            value=dict(source=source,data_sha256=file_sha(data),examples=len(rows),scores=scores,
                       score_definition='Mean target-token NLL under frozen base model; no parameter updates')
            save(ROOT/(source+'.json'),value)
            record['sources'][source]=dict(examples=len(rows),data_sha256=value['data_sha256'],score_sha256=file_sha(ROOT/(source+'.json')))
            save(ROOT/'execution.json',record)
        record.update(status='completed',gpu_seconds=time.monotonic()-started)
        save(ROOT/'execution.json',record)
        save(ROOT/'result.json',record)
        print(json.dumps(record),flush=True)
    except BaseException as error:
        record.update(status='failed',error_type=type(error).__name__,gpu_seconds=time.monotonic()-started)
        save(ROOT/'execution.json',record)
        raise


if __name__=='__main__':
    main()
