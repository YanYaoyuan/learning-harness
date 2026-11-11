#!/usr/bin/env python3
"""Export an explicit, credential-checked research snapshot without touching runs."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
TRACE_NAMES={'steps.jsonl','events.jsonl','calls.jsonl','batches.jsonl'}
SKIP_NAMES={'tokenizer.json','tokenizer_config.json','vocab.json','special_tokens_map.json',
    'added_tokens.json','generation_config.json','chat_template.json','trainer_state.json'}
SAFE_EXT={'.py','.sh','.md','.json','.jsonl','.toml','.txt','.yaml','.yml','.csv','.png','.pdf','.svg'}
TEXT_EXT={'.py','.sh','.md','.json','.jsonl','.toml','.txt','.yaml','.yml','.csv','.svg'}
TOKENS=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{40,}|sk-[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')


def checked(data,name):
    if TOKENS.search(data):raise ValueError('Credential-like value: '+name+' (value suppressed)')


def candidates():
    yield from [ROOT/n for n in ['README.md','pyproject.toml','.env.example']]
    for top in ['src','scripts','tests','docs','requirements','configs']:
        for p in sorted((ROOT/top).rglob('*')):
            if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts:continue
            if p.name.endswith('.local.json') or p.suffix=='.pyc':continue
            if p.suffix in SAFE_EXT or p.name=='lh':yield p
    for p in sorted((ROOT/'data/manifests').rglob('*')):
        if p.is_file() and p.suffix in {'.json','.jsonl'}:yield p
    for top in ['output/figures','output/research-audit','output/baseline-benchmark-v1','output/source-comparison-v1']:
        for p in sorted((ROOT/top).rglob('*')):
            if p.is_file() and not any(x in {'wandb','logs','__pycache__'} for x in p.relative_to(ROOT/top).parts):
                if p.suffix in SAFE_EXT:yield p
    for p in sorted((ROOT/'output/pdf/agent-research').glob('*')):
        if p.is_file() and p.suffix in {'.md','.json','.py'}:yield p
    for relative in ['third_party/ReAct/LICENSE','third_party/ReAct/source.json',
        'third_party/ReAct/prompts/alfworld_3prompts.json','third_party/Reflexion/LICENSE','third_party/Reflexion/source.json',
        'third_party/Reflexion/alfworld_runs/reflexion_few_shot_examples.txt',
        'third_party/Reflexion/alfworld_runs/alfworld_trial.py',
        'third_party/Reflexion/alfworld_runs/generate_reflections.py',
        'third_party/Reflexion/alfworld_runs/env_history.py']:
        yield ROOT/relative
    for p in sorted((ROOT/'runs').rglob('*')):
        if not p.is_file() or p.is_symlink() or p.suffix not in SAFE_EXT or p.name in SKIP_NAMES:continue
        if any(x in {'logs','wandb','__pycache__'} for x in p.parts):continue
        # Running evaluations: snapshot task outcomes/configs, not an inconsistent
        # tail of their larger event streams. Completed collection traces are kept.
        if 'source-comparison-v1' in p.parts and 'evaluation' in p.parts and p.name in TRACE_NAMES:continue
        yield p


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    target=Path(args.out).resolve()
    if target.exists() and any(target.iterdir()):raise ValueError('Use a new empty snapshot directory')
    target.mkdir(parents=True,exist_ok=True);started=datetime.now(timezone.utc).isoformat()
    manifest=[];counts=Counter()
    for p in sorted(set(candidates())):
        rel=p.relative_to(ROOT);destination=target/rel
        if p.is_symlink():raise ValueError('Symlink not allowed: '+str(rel))
        source_size=p.stat().st_size;destination.parent.mkdir(parents=True,exist_ok=True)
        raw_sha=hashlib.sha256();trimmed=False
        if p.name in TRACE_NAMES:
            destination=destination.with_name(destination.name+'.gz')
            with p.open('rb') as inp,destination.open('wb') as out,gzip.GzipFile(fileobj=out,mode='wb',mtime=0) as zipped:
                remaining=source_size
                while remaining>0:
                    line=inp.readline(remaining);remaining-=len(line)
                    if not line:break
                    if not line.endswith(b'\n'):trimmed=True;break
                    checked(line,str(rel));raw_sha.update(line);zipped.write(line)
            counts['compressed_traces']+=1
        else:
            with p.open('rb') as f:data=f.read(source_size)
            if p.suffix in TEXT_EXT or p.name in {'lh','.env.example','LICENSE'}:checked(data,str(rel))
            if p.suffix=='.jsonl' and data and not data.endswith(b'\n'):
                data=data[:data.rfind(b'\n')+1];trimmed=True
            raw_sha.update(data);destination.write_bytes(data)
            if p.stat().st_mode&0o111:destination.chmod(0o755)
        if destination.stat().st_size>45*1024**2:raise ValueError('Oversize Git file: '+str(destination.relative_to(target)))
        manifest.append(dict(path=str(destination.relative_to(target)),source_path=str(rel),
            source_snapshot_sha256=raw_sha.hexdigest(),sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            bytes=destination.stat().st_size,incomplete_last_line_dropped=trimmed))
        counts['files']+=1;counts['bytes']+=destination.stat().st_size
    # Snapshot repository tracks selected research records; future large runtime
    # files and credentials remain ignored. Original workspace ignore is untouched.
    (target/'.gitignore').write_text('''\
.venv*/
.cache/
__pycache__/
*.pyc
.pytest_cache/
.env
.env.*
!.env.example
key.env
*.key.env
.aws/
.codex/
.agents/
configs/*.local.json
data/models/
data/alfworld/
data/downloads/
data/teacher/
*.safetensors
*.pt
*.pth
*.bin
**/tokenizer.json
**/vocab.json
*.log
*.rrd
*.zip
*.egg-info/
**/wandb/
output/wandb/
output/wandb-transfer/
output/pdf/agent-research/papers/
''')
    info=dict(created_started_utc=started,created_finished_utc=datetime.now(timezone.utc).isoformat(),
        client_date='2026-11-11',kind='research snapshot; not a globally atomic backup of active jobs',
        counts=dict(counts),files=manifest,
        exclusions=['credentials and machine settings','model/adapter weights and tokenizer copies','raw ALFWorld/downloaded datasets',
            'downloaded third-party paper PDFs','SDK caches and stdout logs','event streams of still-evolving source-comparison evaluations'],
        notes=['Runtime source code and existing experiment records are copied without modification.',
            'Large trace JSONL files are gzip-compressed losslessly, except any incomplete final line is reported.',
            'No experiment process is stopped or modified. Active progress snapshots may differ by capture time.'])
    (target/'SNAPSHOT_MANIFEST.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(dict(directory=str(target),**dict(counts)),indent=2))


if __name__=='__main__':main()
