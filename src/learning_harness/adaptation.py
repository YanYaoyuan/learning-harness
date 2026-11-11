"""Additive frozen A + fresh trainable B LoRA, without BF16 weight merging."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .common import digest, file_sha


def adapter_config(config):
    from peft import LoraConfig
    return LoraConfig(r=config['lora_r'],lora_alpha=config['lora_alpha'],lora_dropout=config['lora_dropout'],
        target_modules=config['target_modules'],task_type='CAUSAL_LM',bias='none')


def parameter_hash(model,adapter):
    import torch
    h=hashlib.sha256()
    for name,p in sorted(model.named_parameters()):
        if 'lora_' in name and f'.{adapter}.' in name:
            h.update(name.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def freeze_a_dropout(model):
    for module in model.modules():
        if hasattr(module,'lora_dropout') and 'A' in module.lora_dropout:module.lora_dropout['A'].eval()


def build(model_path,revision,config,a_adapter=None,b_adapter=None,seed=1101,training=False,check_inputs=None):
    import torch
    from peft import get_peft_model,PeftModel
    from transformers import AutoModelForCausalLM,set_seed
    base=AutoModelForCausalLM.from_pretrained(model_path,revision=revision,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',trust_remote_code=False).to('cuda:0').eval()
    base.config.use_cache=not training

    def logits(m):
        with torch.inference_mode():return m(**check_inputs,logits_to_keep=8).logits.detach().clone()

    original=logits(base) if check_inputs is not None and a_adapter is None else None
    if a_adapter:
        model=PeftModel.from_pretrained(base,str(a_adapter),adapter_name='A',is_trainable=False)
        loaded=model.peft_config['A']
        assert loaded.r==config['lora_r'] and loaded.lora_alpha==config['lora_alpha']
        assert set(loaded.target_modules)==set(config['target_modules'])
    else:
        set_seed(0)
        model=get_peft_model(base,adapter_config(config),adapter_name='A')
    model.eval()
    before=logits(model) if check_inputs is not None else None
    if original is not None and not torch.equal(original,before):
        raise ValueError('Zero A branch changed the base logits')
    set_seed(seed)
    if b_adapter:model.load_adapter(str(b_adapter),adapter_name='B',is_trainable=training)
    else:model.add_adapter('B',adapter_config(config))
    model.base_model.set_adapter(['A','B'])
    for name,p in model.named_parameters():p.requires_grad_(training and 'lora_' in name and '.B.' in name)
    model.eval()
    if before is not None:
        if b_adapter is not None:raise ValueError('Step-zero checks require a new B adapter')
        after=logits(model)
        if not torch.equal(before,after):raise ValueError('Fresh zero B changed A logits')
    info=dict(a_parameter_sha256=parameter_hash(model,'A'),b_initial_parameter_sha256=parameter_hash(model,'B'),
        exact_zero_b_logit_check=check_inputs is not None,
        exact_zero_a_logit_check=original is not None,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        trainable_scope='B adapter only' if training else 'none')
    return model,info


def composition_revision(path):
    path=Path(path);meta=json.loads((path/'composition.json').read_text())
    if meta['a_adapter'] is not None:
        assert file_sha(Path(meta['a_adapter'])/'adapter_model.safetensors')==meta['a_adapter_sha256']
    b=path/'B'
    assert file_sha(b/'adapter_model.safetensors')==meta['b_adapter_sha256']
    identity=dict(base_revision=meta['base_revision'],a=meta['a_adapter_sha256'] or 'zero',
        b=meta['b_adapter_sha256'],lora_config=meta['lora_config'])
    return meta['base_revision']+'/composition-sha256:'+digest(identity)


def load_composition(path,model_path,revision):
    path=Path(path);meta=json.loads((path/'composition.json').read_text())
    served=composition_revision(path)
    if meta['base_revision']!=revision:raise ValueError('Base revision mismatch')
    model,_=build(model_path,revision,meta['lora_config'],a_adapter=meta['a_adapter'],
                  b_adapter=path/'B',training=False)
    return model,served


def save_composition(model,path,model_path,revision,config,a_adapter,step,extra=None):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(path,selected_adapters=['B'],safe_serialization=True)
    a_path=str(Path(a_adapter).absolute()) if a_adapter else None
    meta=dict(base_model=model_path,base_revision=revision,lora_config=config,
        a_adapter=a_path,a_adapter_sha256=file_sha(Path(a_path)/'adapter_model.safetensors') if a_path else None,
        b_adapter_sha256=file_sha(path/'B/adapter_model.safetensors'),optimizer_step=step,
        semantics='frozen A plus additive B; no merging',**(extra or {}))
    (path/'composition.json').write_text(json.dumps(meta,indent=2)+'\n')
    return composition_revision(path)
