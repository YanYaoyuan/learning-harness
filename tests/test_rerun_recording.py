from pathlib import Path

import pytest

from learning_harness.rerun_recording import LineCursor, final_scores, loss_point


def test_tail_waits_for_complete_utf8_line_and_does_not_duplicate(tmp_path):
    p=tmp_path/'steps.jsonl';cursor=LineCursor(p)
    assert list(cursor.read())==[]
    value='冷却反馈'.encode('utf8')
    p.write_bytes(b'first\n'+value[:4])
    assert list(cursor.read())==['first\n']
    assert list(cursor.read())==[]
    with p.open('ab') as stream:stream.write(value[4:]+b'\n')
    assert list(cursor.read())==['冷却反馈\n']
    assert list(cursor.read())==[]


def test_tail_rejects_truncation_instead_of_double_counting(tmp_path):
    p=tmp_path/'steps.jsonl';p.write_text('old completed line\n')
    cursor=LineCursor(p);assert len(list(cursor.read()))==1
    p.write_text('new\n')
    with pytest.raises(RuntimeError,match='replaced/truncated'):list(cursor.read())


def test_checkpoint_only_emitted_for_complete_unique_expected_episodes():
    meta=dict(planned_episodes=3,config=dict(arms=['none'],rollout_seeds=[11]))
    manifest=[dict(task_id=t,evaluation_role=role) for t,role in [('one','B_query'),('two','B_query'),('three','A_retention')]]
    rows=[dict(task_id=t['task_id'],arm='none',rollout_seed=11,success=i!=1,status='completed') for i,t in enumerate(manifest)]
    assert final_scores(meta,rows[:2],manifest) is None
    result=final_scores(meta,rows,manifest)
    assert result['B']['success_pct']==50 and result['B']['tasks']==2
    assert result['A']['success_pct']==100 and result['A']['tasks']==1
    with pytest.raises(ValueError,match='duplicated'):final_scores(meta,[rows[0],rows[0],rows[2]],manifest)
    with pytest.raises(ValueError,match='Failed'):final_scores(meta,[dict(r,status='error') for r in rows],manifest)


def test_training_step_comes_from_known_cadence_not_rounded_epoch():
    line="\r 12%|progress| {'loss': 0.3, 'learning_rate': 0.0001, 'grad_norm': 1.5, 'epoch': 0.12}\n"
    point=loss_point(line,index=1,cadence=8,max_steps=128)
    assert point['optimizer_step']==16  # round(.12*128) would incorrectly give 15
    assert point['loss']==.3
    assert loss_point("{'train_loss': 0.3}\n",0,8,128) is None
    with pytest.raises(ValueError,match='cadence'):loss_point(line,16,8,128)
