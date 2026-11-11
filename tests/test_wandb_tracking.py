import importlib.util
from pathlib import Path
import pytest


spec=importlib.util.spec_from_file_location('track_wandb','scripts/track_wandb.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_trace_export_uses_explicit_public_fields():
    row=dict(task_id='public-task',action='look',observation='public feedback',step=0,
        api_key='must-not-upload',headers={'Authorization':'must-not-upload'},
        messages=[{'role':'system','content':'system instructions'},
            {'role':'user','content':'public room and goal'},
            {'role':'user','content':'[TRAINING SUPPORT] public hint'}])
    exported=module.trace_row(row)
    assert 'api_key' not in exported and 'headers' not in exported and 'messages' not in exported
    assert exported['public_initial']=='public room and goal'
    assert exported['visible_support']=='[TRAINING SUPPORT] public hint'


def test_collection_and_two_evaluation_layouts_map_to_correct_cloud_runs():
    assert module.case_for(Path('evaluation/main/filtered-none-seed1101/step-0032'))==('filtered-none-seed1101','adaptation',32)
    assert module.case_for(Path('evaluation/api-seed1202/B/step-0016'))==('api-seed1202','adaptation',16)
    assert module.case_for(Path('collection/generic-shard1'))==('generic','collection',None)


def test_empty_collection_has_no_estimate_and_nonempty_counts_are_exact():
    assert module.collection_values([]) is None
    rows=[dict(task_id='t',rollout_seed=seed,success=win,input_tokens=10,output_tokens=2)
          for seed,win in [(11,1),(23,0)]]
    assert module.collection_values(rows)==dict(completed_episodes=2,success_pct=50,
        wins=1,input_tokens=20,output_tokens=4)
    with pytest.raises(ValueError,match='Duplicate'):
        module.collection_values(rows+rows[:1])


def test_family_studies_have_distinct_cloud_identifiers():
    assert module.study_name(Path('runs/transfer-factorial-v1/cool'))=='transfer-factorial-v1-cool'
    assert module.study_name(Path('runs/transfer-factorial-v1/heat'))=='transfer-factorial-v1-heat'
    assert module.study_name(Path('runs/api-harness-v1'))=='api-harness-v1'
