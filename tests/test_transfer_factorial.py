import importlib.util
from pathlib import Path
import pytest
from learning_harness.reference_heating import heating_plan


def test_heating_reference_obeys_public_discovery_and_door_feedback():
    p=heating_plan('You see a countertop 1, a microwave 1, a cabinet 1.\nYour task is to: put a hot apple in cabinet.')
    assert next(p)=='go to countertop 1'
    assert p.send('On the countertop 1, you see an apple 2.')=='take apple 2 from countertop 1'
    assert p.send('You pick up the apple 2.')=='go to microwave 1'
    assert p.send('The microwave 1 is closed.')=='open microwave 1'
    assert p.send('You open it.')=='heat apple 2 with microwave 1'
    assert p.send('Nothing happens.')=='close microwave 1'
    assert p.send('You close it.')=='heat apple 2 with microwave 1'
    assert p.send('You heat it.')=='go to cabinet 1'
    assert p.send('The cabinet 1 is closed.')=='open cabinet 1'
    assert p.send('You open it.')=='move apple 2 to cabinet 1'


def test_reference_collection_rejects_query_before_environment_creation():
    spec=importlib.util.spec_from_file_location('transfer_reference','scripts/collect_transfer_reference.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for role in ['b_query','b_calibration','a_retention']:
        with pytest.raises(ValueError,match='Only support or diagnostic'):
            m.episode((dict(partition='dev',adaptation_role=role),120))


def test_frozen_transfer_groups_are_disjoint_and_final_families_preserved():
    import json
    from learning_harness.common import read_jsonl
    root=Path('runs/transfer-factorial-v1')
    if not (root/'design.json').exists():pytest.skip('Local frozen study not present')
    groups=[]
    for family in ['cool','heat']:
        for role in ['b_support_candidates','b_calibration','b_diagnostic','b_query']:
            rows=read_jsonl(root/family/f'{role}.jsonl')
            assert all(r['official_split']=='train' for r in rows)
            assert len({r['task_group'] for r in rows})==len(rows)
            groups.append({r['task_group'] for r in rows})
    groups.append({r['task_group'] for r in read_jsonl(root/'a_query.jsonl')})
    assert sum(map(len,groups))==len(set.union(*groups))
    for f in ['cool','heat']:
        d=json.loads((root/f/'design.json').read_text())
        assert d['family'] not in ['pick_two_obj_and_place','look_at_obj_in_light']


def test_source_selection_interaction_is_paired_difference_of_differences():
    import sys
    import numpy as np
    sys.path.insert(0,str(Path('scripts').absolute()))
    from transfer_factorial_report import contrasts
    values={'matched-generic':np.array([[.2,.4]]),'matched-api':np.array([[.3,.5]]),
            'full-generic':np.array([[.3,.2]]),'full-api':np.array([[.6,.4]])}
    result=contrasts(values)
    assert np.allclose(result['interaction: full - matched'],[[.2,.1]])
