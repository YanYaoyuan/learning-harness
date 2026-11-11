import importlib.util
from pathlib import Path
import sys

import numpy as np


def module():
    scripts=str(Path('scripts').absolute())
    sys.path.insert(0,scripts)
    try:
        spec=importlib.util.spec_from_file_location('baseline_report','scripts/baseline_benchmark_report.py')
        result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
        return result
    finally:sys.path.remove(scripts)


def test_cluster_pairing_keeps_all_seeds_in_task_group():
    m=module()
    # Multiple seeds in one task group must not manufacture independent groups.
    a=[dict(task_id='one',task_group='g1',seed=s,success=True) for s in [11,23,37]]
    b=[dict(r,success=False) for r in a]
    result=m.paired_effect(a,b,'success')
    assert result['task_groups']==1 and result['paired_task_seeds']==3
    assert result['difference_pp']==100 and result['ci95_pp']==[100,100]
    assert m.paired_effect(a,b[:-1],'success') is None


def test_zero_success_has_nonzero_uncertainty():
    m=module()
    lower,upper=m.wilson(0,40)
    assert abs(lower)<1e-12 and .08<upper<.10


def test_demo_signature_ignores_container_order_but_not_goal():
    scripts=str(Path('scripts').absolute());sys.path.insert(0,scripts)
    try:
        from audit_baseline_prompts import signature
        assert signature('a cabinet 1 and a table 2','put cup on table')==signature('a table 2 and a cabinet 1','put cup on table')
        assert signature('a table 2','put cup on table')!=signature('a table 2','put cup in cabinet')
    finally:sys.path.remove(scripts)
