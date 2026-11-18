#!/usr/bin/env python3
"""Download official small data and test CPU text mechanics on synthetic goals only."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import urllib.request
import shutil
import argparse

SOURCE=Path('third_party/WebShop').absolute()
OUT=Path('output/webshop-preparation/20261112')
DATA=Path('/media/user/50D2BD30D2BD1AE4/WorkSpace/learning-harness-lab/data/webshop/smoke1000')
FILES=[('items_shuffle_1000.json','1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib',list),
       ('items_ins_v2_1000.json','1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu',dict),
       ('items_human_ins.json','14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O',dict)]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mirrored_file(name,target,kind):
    repositories=[('YWZBrandon/webshop-data',''),('HongbangYuan/webshop',''),('sparklabutah/timewarp-env-data','webshop/')]
    copies=[]
    for repo,prefix in repositories:
        with urllib.request.urlopen('https://huggingface.co/api/datasets/'+repo,timeout=30) as response:
            metadata=json.load(response)
        path=prefix+name
        if path not in {v['rfilename'] for v in metadata.get('siblings',[])}:continue
        revision=metadata['sha'];url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}'
        temporary=target.with_name(target.name+f'.mirror{len(copies)}.part')
        if temporary.exists():raise ValueError('Partial mirror copy retained for inspection')
        with urllib.request.urlopen(url,timeout=45) as response,temporary.open('xb') as output:
            shutil.copyfileobj(response,output)
        value=json.loads(temporary.read_text())
        if not isinstance(value,kind) or not value:raise ValueError('Mirror schema mismatch')
        copies.append(dict(repository=repo,revision=revision,url=url,sha256=sha(temporary),entries=len(value),temporary=str(temporary)))
        if len(copies)==2:break
    if len(copies)!=2 or copies[0]['sha256']!=copies[1]['sha256']:
        raise ValueError('Independent mirrors unavailable or not byte-identical')
    Path(copies[0]['temporary']).replace(target)
    Path(copies[1]['temporary']).unlink()
    for record in copies:record.pop('temporary')
    return copies


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--community-mirror',action='store_true')
    args=parser.parse_args()
    import gdown
    import numpy as np
    import torch
    if json.loads((OUT/'runtime.json').read_text())['status']!='ready':raise ValueError('CPU runtime not ready')
    receipt=OUT/'smoke.json'
    if receipt.exists():raise ValueError('Smoke record exists; preserve before another attempt')
    state=dict(status='running',started_unix=time.time(),smoke_only=True,product_catalog='official 1000-product subset',
        human_goals_used=False,official_test_evaluations=0,model_calls=0,gpu_used=False,data=[])
    if args.community_mirror:
        state.update(product_catalog='1000-product community mirror preview',original_drive_download_failed=True,
                     original_author_byte_identity_verified=False,independent_mirrors_cross_checked=True)
    def save():
        tmp=receipt.with_suffix('.json.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(receipt)
    save()
    try:
        DATA.mkdir(parents=True,exist_ok=True);(SOURCE/'data').mkdir(exist_ok=True)
        for name,identifier,kind in FILES:
            target=DATA/name
            if target.exists():raise ValueError('Existing dataset file lacks this run provenance: '+str(target))
            temporary=target.with_suffix('.json.part')
            if temporary.exists():raise ValueError('Partial dataset retained for inspection')
            mirrors=mirrored_file(name,target,kind) if args.community_mirror else None
            if not mirrors:gdown.download(id=identifier,output=str(temporary),quiet=False)
            value=json.loads((target if mirrors else temporary).read_text())
            if not isinstance(value,kind) or not value:raise ValueError('Official dataset schema mismatch')
            if name=='items_shuffle_1000.json' and len(value)!=1000:raise ValueError('Preview catalog size mismatch')
            if not mirrors:temporary.replace(target)
            link=SOURCE/'data'/name
            if link.exists() or link.is_symlink():raise ValueError('Source data link already exists')
            link.symlink_to(target)
            state['data'].append(dict(name=name,google_drive_id=identifier,bytes=target.stat().st_size,
                sha256=sha(target),entries=len(value),path=str(target),community_mirrors=mirrors))
            save()
        critical=[SOURCE/p for p in ['web_agent_site/engine/engine.py','web_agent_site/engine/goal.py',
                                   'web_agent_site/envs/web_agent_text_env.py','web_agent_site/utils.py']]
        before={str(p.relative_to(SOURCE)):sha(p) for p in critical}
        sys.path.insert(0,str(SOURCE))
        from web_agent_site.engine.engine import load_products
        random.seed(11);np.random.seed(11);torch.manual_seed(11)
        products,*_=load_products(filepath=str(SOURCE/'data/items_shuffle_1000.json'),num_products=1000,human_goals=False)
        resources=SOURCE/'search_engine/resources_1k';resources.mkdir(parents=True,exist_ok=True)
        with (resources/'documents.jsonl').open('x') as handle:
            for p in products:
                options=', and '.join(f"{k}: {', '.join(v)}" for k,v in p.get('options',{}).items())
                contents=' '.join([p['Title'],p['Description'],p['BulletPoints'][0],options]).lower()
                handle.write(json.dumps(dict(id=p['asin'],contents=contents,product=p))+'\n')
        index=SOURCE/'search_engine/indexes_1k'
        args=[sys.executable,'-m','pyserini.index.lucene','--collection','JsonCollection','--input',str(resources),
            '--index',str(index),'--generator','DefaultLuceneDocumentGenerator','--threads','1',
            '--storePositions','--storeDocvectors','--storeRaw']
        with (OUT/'small-index.log').open('x') as log:
            subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        random.seed(11);np.random.seed(11);torch.manual_seed(11)
        env=WebAgentTextEnv(observation_mode='text',num_products=1000,human_goals=False,get_image=False)
        observation,_=env.reset(session=0)
        if not isinstance(observation,str) or not observation:raise ValueError('No text observation')
        public_instruction=env.get_instruction_text()
        steps=[];candidates=[]
        for query in ['shirt',' '.join(public_instruction.split()[:8]),' '.join(public_instruction.split()[:4])]:
            action='search['+query+']';ob,reward,done,_=env.step(action)
            steps.append(dict(action=action,observation_nonempty=bool(ob),reward=float(reward),done=done))
            candidates=[s for s in env.get_available_actions()['clickables'] if re.fullmatch('[a-z0-9]{10}',s)]
            if candidates:break
        if not candidates:raise ValueError('No visible search results after public queries')
        for action in ['click['+candidates[0]+']','click[buy now]']:
            ob,reward,done,_=env.step(action)
            if not math.isfinite(float(reward)):raise ValueError('Nonfinite environment reward')
            steps.append(dict(action=action,observation_nonempty=bool(ob),reward=float(reward),done=done))
        if not done:raise ValueError('Synthetic purchase did not terminate')
        after={str(p.relative_to(SOURCE)):sha(p) for p in critical}
        if before!=after:raise ValueError('Upstream environment code changed during smoke')
        state.update(status='ready_for_development',finished_unix=time.time(),products_loaded=len(products),
            index_documents_sha256=sha(resources/'documents.jsonl'),steps=steps,upstream_code_sha256=before,
            torch_version=torch.__version__,torch_cuda_available=torch.cuda.is_available(),
            full_benchmark_ready=False,readiness_only_not_an_agent_performance_result=True)
        save();print(json.dumps(state),flush=True)
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error),finished_unix=time.time());save();raise


if __name__=='__main__':main()
