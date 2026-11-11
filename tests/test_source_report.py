import importlib.util
import numpy as np


def test_paired_auc_separates_prior_advantage_from_update_gain():
    spec=importlib.util.spec_from_file_location('source_report','scripts/source_comparison_report.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    # A constant starting advantage changes raw AUC, not gain AUC.
    a={f:np.ones((3,4,5)) for f in ['cool','heat']}
    b={f:np.zeros((3,4,5)) for f in ['cool','heat']}
    result=m.paired_bootstrap(a,b)
    assert result['raw_auc']['difference_pp']==100
    assert result['raw_auc']['ci95_pp']==[100,100]
    assert result['gain_auc']['difference_pp']==0
    assert result['gain_auc']['ci95_pp']==[0,0]


def test_opposite_family_effects_cancel_only_in_the_equal_weight_mean():
    spec=importlib.util.spec_from_file_location('source_report','scripts/source_comparison_report.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    a={'cool':np.ones((3,4,5)),'heat':np.zeros((3,4,7))}
    b={'cool':np.zeros((3,4,5)),'heat':np.ones((3,4,7))}
    result=m.paired_bootstrap(a,b)['raw_auc']
    assert result['difference_pp']==0 and result['family_differences_pp']=={'cool':100,'heat':-100}
