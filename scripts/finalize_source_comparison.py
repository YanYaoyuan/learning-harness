#!/usr/bin/env python3
"""Finish local reports even when the independent cloud observer disconnects."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('runs/source-comparison-v1');OUT=Path('output/source-comparison-v1')


def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    while True:
        path=ROOT/'execution.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        if state.get('status')=='completed':
            env=dict(os.environ,PYTHONPATH=str(Path('src').absolute()),MPLCONFIGDIR='/tmp/lh-mpl')
            with (ROOT/'local-finalization-report.log').open('w') as handle:
                subprocess.run(['.venv-model/bin/python','scripts/source_comparison_report.py'],
                    env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
            result=json.loads((OUT/'summary.json').read_text())
            if not result['complete'] or not result['token_and_budget_audit_passed']:raise ValueError('Incomplete report')
            (OUT/'finalization.json').write_text(json.dumps(dict(complete=True,
                evaluation_task_runs=result['completed_evaluation_task_runs'],token_and_budget_audit_passed=True,
                official_final_tests_used=False),indent=2)+'\n')
            print('Local measured report and figures completed.',flush=True);return
        if state.get('status') in ['failed','matching_gate_failed']:
            print('Pipeline stopped: '+state['status']+'; raw records retained.',flush=True);return
        if time.monotonic()-started>24*3600:raise TimeoutError('Pipeline did not finish in 24 hours')
        time.sleep(30)


if __name__=='__main__':main()
