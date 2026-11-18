#!/usr/bin/env python3
"""Download pinned official 2022 source/LG NLP model and run its unchanged CPU tests."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

REVISION = '61e56dbba107ae13ff49e6058d7d2aad23326929'
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/webshop-preparation/20261112/original-reference'
SOURCE = ROOT/'third_party/WebShop-2022-reference'
STORE = Path('/media/user/50D2BD30D2BD1AE4/WorkSpace/learning-harness-lab/data/models/webshop-nlp')
PYTHON = ROOT/'.venv-webshop/bin/python'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def download(url, destination):
    if destination.exists() or destination.with_suffix(destination.suffix+'.part').exists():
        raise ValueError('Preserve existing download before another attempt')
    partial = destination.with_suffix(destination.suffix+'.part')
    with urllib.request.urlopen(url, timeout=60) as response, partial.open('xb') as handle:
        expected = response.headers.get('Content-Length')
        shutil.copyfileobj(response, handle, length=1024*1024)
        handle.flush()
        os.fsync(handle.fileno())
    if expected is not None and partial.stat().st_size != int(expected):
        raise ValueError('Truncated download')
    partial.replace(destination)
    return dict(url=url, path=str(destination), bytes=destination.stat().st_size, sha256=sha(destination))


def main():
    if OUT.exists() or SOURCE.exists():
        raise ValueError('Preserve existing reference attempt before restarting')
    OUT.mkdir(parents=True)
    STORE.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(STORE).free < 2*1024**3:
        raise ValueError('Need 2 GiB on user-specified storage disk')
    state = dict(status='preparing', started_unix=time.time(), source_revision=REVISION,
                 gpu_used=False, model_calls=0, official_test_evaluations=0,
                 original_current_source_modified=False)
    receipt = OUT/'reference.json'
    def save():
        p = receipt.with_suffix('.json.tmp')
        p.write_text(json.dumps(state, indent=2)+'\n')
        p.replace(receipt)
    save()
    try:
        archive = ROOT/'data/downloads'/f'webshop-{REVISION}.tar.gz'
        state['source_archive'] = download(f'https://codeload.github.com/princeton-nlp/WebShop/tar.gz/{REVISION}', archive)
        save()
        SOURCE.mkdir()
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts
                if len(parts) < 2 or '..' in parts or Path(member.name).is_absolute():
                    if member.isdir():
                        continue
                    raise ValueError('Unsafe source archive path')
                if not member.isfile():
                    continue
                path = SOURCE.joinpath(*parts[1:])
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as inp, path.open('xb') as dest:
                    shutil.copyfileobj(inp, dest)
        old_goal = SOURCE/'web_agent_site/engine/goal.py'
        current_goal = ROOT/'third_party/WebShop/web_agent_site/engine/goal.py'
        if old_goal.read_bytes() != current_goal.read_bytes().replace(b'en_core_web_sm', b'en_core_web_lg'):
            raise ValueError('Unexpected reward differences beyond official model switch')
        state['original_goal_sha256'] = sha(old_goal)
        state['current_goal_sha256'] = sha(current_goal)
        state['reward_logic_identical_except_model_name'] = True
        state['test_sha256'] = {}
        for name in ('test_goal.py', 'test_normalize.py'):
            relative = Path('tests/web-agent-site/engine')/name
            if sha(SOURCE/relative) != sha(ROOT/'third_party/WebShop'/relative):
                raise ValueError('Original and current test files differ')
            state['test_sha256'][str(relative)] = sha(SOURCE/relative)
        save()
        wheel = STORE/'en_core_web_lg-3.3.0-py3-none-any.whl'
        state['nlp_wheel'] = download('https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.3.0/en_core_web_lg-3.3.0-py3-none-any.whl', wheel)
        save()
        target = STORE/'python-packages'
        env = dict(os.environ, PYTHONPATH='', PIP_CONFIG_FILE=os.devnull, PIP_NO_CACHE_DIR='1',
                   CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
        for key in list(env):
            if key.startswith('LH_TEACHER_'):
                env.pop(key)
        with (OUT/'model-install.log').open('x') as log:
            subprocess.run([str(PYTHON), '-m', 'pip', 'install', '--no-deps', '--target', str(target), str(wheel)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        available_kb = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                if line.startswith('MemAvailable:')))
        if available_kb < 4*1024**2:
            raise ValueError('Need 4 GiB available RAM before CPU NLP checks')
        env['PYTHONPATH'] = str(SOURCE)+os.pathsep+str(target)
        tests = [str(SOURCE/'tests/web-agent-site/engine'/name) for name in ('test_goal.py','test_normalize.py')]
        argv = [str(PYTHON), '-m', 'pytest', '-q', '-o', 'addopts=', '-p', 'no:cacheprovider',
                '--junitxml='+str(OUT/'tests.xml'), *tests]
        with (OUT/'tests.log').open('x') as log:
            result = subprocess.run(argv, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        state.update(status='reference_tests_passed' if result.returncode == 0 else 'reference_tests_failed',
                     tests_returncode=result.returncode, tests_argv=argv,
                     source_path=str(SOURCE), nlp_pythonpath=str(target), ended_unix=time.time(),
                     full_benchmark_ready=False)
        save()
        print(json.dumps(state), flush=True)
        if result.returncode:
            raise RuntimeError('Original reward tests failed; inspect retained log')
    except BaseException as error:
        state.update(error_type=type(error).__name__, error=str(error), ended_unix=time.time())
        if state['status'] == 'preparing':
            state['status'] = 'failed'
        save()
        raise


if __name__ == '__main__':
    main()
