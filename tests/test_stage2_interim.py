from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from stage2_interim_report import analyze


def row(init,sample,source,arm,value):
    return dict(case=dict(initialization_seed=init,sampling_seed=sample,source=source,arm=arm,repeat=0),
        mean_success=value,mean_nll=1-value,success=dict(cool=value,heat=value),
        exposure=dict(input_tokens=100,target_tokens=10))


def test_interim_uses_all_complete_quadruplets_and_excludes_unpaired_scores():
    records=[]
    for sample in (101,202):
        for source in ("api","generic"):
            for arm in ("full_pass","subset_replay"):
                records.append(row(101,sample,source,arm,.3 if arm=="full_pass" else .2))
    # A high unpaired value cannot improve the matched mean.
    records.append(row(202,101,"api","full_pass",1))
    result=analyze(dict(records=records,snapshot_china="test",freeze_sha256="test"))
    assert result["completed_models"]==9
    assert result["matched_cells"]==2
    assert result["matched_models"]==8
    assert result["results"]["api"]["policy_effect"]["success_difference_pp"]==pytest.approx(10)
    assert not result["conditional_sampling"][0]["complete_four_seed_row"]


def test_conditional_sampling_spread_is_not_pooled_across_initializations():
    records=[]
    for init in (101,202):
        for sample in (101,202,303,404):
            for source in ("api","generic"):
                for arm in ("full_pass","subset_replay"):
                    value=.2 if source=="generic" else (.3 if init==101 else .7)
                    records.append(row(init,sample,source,arm,value))
    result=analyze(dict(records=records,snapshot_china="test",freeze_sha256="test"))
    assert len(result["conditional_sampling"])==2
    for conditional in result["conditional_sampling"]:
        assert conditional["complete_four_seed_row"]
        assert conditional["full_pass"]["sampling_sd_pp"]==0
