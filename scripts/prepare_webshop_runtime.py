#!/usr/bin/env python3
"""Build an isolated CPU-only WebShop runtime; never invoke upstream setup.sh."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('output/webshop-preparation/20261112')
VENV=Path('.venv-webshop')


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'runtime.json'
    if path.exists():raise ValueError('Preparation record exists; inspect before another attempt')
    env=dict(os.environ,PYTHONPATH='',PIP_CONFIG_FILE=os.devnull,PIP_REQUIRE_VIRTUALENV='true')
    env.pop('PIP_USER',None)
    for key in list(env):
        if key.startswith('LH_TEACHER_'):env.pop(key)
    record=dict(status='running',started_unix=time.time(),steps=[],gpu_used=False,
        existing_training_environments_modified=False,upstream_setup_script_executed=False)
    def save():
        temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)
    save()
    python=str((VENV/'bin/python').absolute())
    commands=[
        ('venv',[sys.executable,'-m','venv',str(VENV)]),
        ('pip',[python,'-m','pip','install','--index-url','https://pypi.org/simple','pip==23.3.2','setuptools==65.5.1','wheel==0.41.3']),
        ('cpu-torch',[python,'-m','pip','install','--index-url','https://download.pytorch.org/whl/cpu','--no-deps','torch==1.11.0+cpu']),
        ('runtime',[python,'-m','pip','install','--index-url','https://pypi.org/simple','--prefer-binary',
                    '--only-binary=numpy,pandas,scipy,scikit-learn,spacy,thinc,blis,tokenizers,pyjnius',
                    '-r','configs/webshop-runtime-requirements.txt']),
        ('pyserini',[python,'-m','pip','install','--index-url','https://pypi.org/simple','--no-deps','pyserini==0.17.0']),
        ('validation',[python,'-m','pip','install','--index-url','https://pypi.org/simple',
                       '-r','configs/webshop-validation-requirements.txt']),
        ('spacy-model',[python,'-m','pip','install','--no-deps',
             'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.3.0/en_core_web_sm-3.3.0-py3-none-any.whl']),
        ('import-check',[python,'-c',
             "import sys;sys.path.insert(0,'third_party/WebShop');import torch,spacy;from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv;print('IMPORT_OK',torch.__version__,spacy.__version__)"])]
    try:
        for name,argv in commands:
            start=time.monotonic()
            with (ROOT/(name+'.log')).open('x') as log:
                result=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
            record['steps'].append(dict(name=name,returncode=result.returncode,seconds=time.monotonic()-start))
            save()
            if result.returncode:raise RuntimeError('Runtime preparation failed at '+name)
        record.update(status='ready',finished_unix=time.time(),python=python)
        save();print(json.dumps(record),flush=True)
    except BaseException as error:
        record.update(status='failed',error_type=type(error).__name__,error=str(error),finished_unix=time.time())
        save();raise


if __name__=='__main__':main()
