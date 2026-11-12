#!/usr/bin/env python3
"""Run upstream handcrafted reward/normalization checks in the isolated CPU venv."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'third_party/WebShop'
    output = root / 'output/webshop-preparation/20261112'
    receipt = output / 'upstream-tests.json'
    if receipt.exists():
        raise ValueError('Preserve the existing test receipt before another attempt')
    tests = [source / 'tests/web-agent-site/engine' / name
             for name in ('test_goal.py', 'test_normalize.py')]
    paths = tests + [source / 'web_agent_site/engine' / name
                     for name in ('goal.py', 'normalize.py')]
    def hashes():
        return {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths}
    before = hashes()
    env = dict(os.environ, PYTHONPATH=str(source), CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    for key in list(env):
        if key.startswith('LH_TEACHER_'):
            env.pop(key)
    python = str(root / '.venv-webshop/bin/python')
    started = time.time()
    argv = [python, '-m', 'pytest', '-q', '-o', 'addopts=', '-p', 'no:cacheprovider',
            '--junitxml=' + str(output / 'upstream-tests.xml'), *map(str, tests)]
    with (output / 'upstream-tests.log').open('x') as log:
        result = subprocess.run(argv, cwd=source, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=180)
    after = hashes()
    record = dict(status='passed' if result.returncode == 0 and before == after else 'failed',
                  returncode=result.returncode, started_unix=started, ended_unix=time.time(),
                  argv=argv, upstream_code_unchanged=before == after,
                  source_sha256=before, official_test_evaluations=0, model_calls=0,
                  gpu_used=False, scope='Upstream handcrafted unit cases only; no human goals')
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    if record['status'] != 'passed':
        raise RuntimeError('Inspect upstream-tests.log')


if __name__ == '__main__':
    main()
