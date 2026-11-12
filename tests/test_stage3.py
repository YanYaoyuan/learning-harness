from pathlib import Path
import signal
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_lora_fixed_schedule import bind_schedule
import run_stage3_task_controls as study
from stage3_task_report import analyze


def test_external_schedule_binding_checks_all_controls():
    recipe=dict(sampling_seed=101,eligible_examples=732,indices=[7,11,19,23,29])
    schedule=bind_schedule(recipe,101)
    assert schedule(732,4,303)==[7,11,19,23]
    with pytest.raises(ValueError):bind_schedule(recipe,202)
    with pytest.raises(ValueError):schedule(731,4,101)
    with pytest.raises(ValueError):schedule(732,6,101)


def test_seventy_model_plan_balances_hardware_and_reference_checks():
    cases=study.plan()
    assert len(cases)==len({c['name'] for c in cases})==70
    assert sum(c['gpu']==0 for c in cases)==35
    assert all(c['arm']=='full_reference' and c['repeat']==2 for c in cases[:2])
    for arm in study.ARMS:
        group=[c for c in cases if c['arm']==arm]
        assert len(group)==34
        assert sum(c['repeat']==0 for c in group)==32
        for c in group:
            mate=next(x for x in cases if x['arm'] in study.ARMS and x['arm']!=arm and
                      all(c[k]==x[k] for k in ('source','initialization_seed','sampling_seed','repeat')))
            assert mate['gpu']==c['gpu']


def test_nll_reductions_and_success_increases_are_counted_as_wins():
    rows=[]
    for arm,value in [('full_reference',.2),('random',.3),('min_nll',.4)]:
        for source in ['api','generic']:
            for init in [101,202,303,404]:
                for sample in [101,202,303,404]:
                    rows.append(dict(arm=arm,source=source,initialization_seed=init,sampling_seed=sample,
                                     repeat=0,mean_success=value,mean_nll=1-value))
    _,_,effects=analyze(rows)
    assert effects['mean_success']['nll_minus_random']['api']['wins']==16
    assert effects['mean_nll']['nll_minus_random']['api']['wins']==16
    assert effects['mean_nll']['nll_minus_random']['api']['mean_difference']==pytest.approx(-.1)
