import pytest

from learning_harness.reference_cooling import cooling_plan,parse_goal


def test_reference_waits_for_public_object_and_respects_door_feedback():
    initial='You see a cabinet 1, a fridge 1, a microwave 1.\nYour task is to: cool some bowl and put it in microwave.'
    p=cooling_plan(initial)
    assert next(p)=='go to microwave 1'
    assert p.send('The microwave 1 is closed.')=='open microwave 1'
    assert p.send('On the microwave 1, you see a bowl 2.')=='take bowl 2 from microwave 1'
    assert p.send('You pick up the bowl 2.')=='go to fridge 1'
    assert p.send('The fridge 1 is closed.')=='open fridge 1'
    assert p.send('You open the fridge 1.')=='cool bowl 2 with fridge 1'
    assert p.send('You cool the bowl 2.')=='go to microwave 1'
    assert p.send('On the microwave 1 you see nothing.')=='move bowl 2 to microwave 1'


def test_reference_does_not_invent_an_unobserved_fridge():
    p=cooling_plan('You see a cabinet 1.\nYour task is to: cool some bowl and put it in cabinet.')
    with pytest.raises(ValueError,match='absent'):next(p)
    assert parse_goal('Your task is to: put a cool apple in countertop.')==('apple','countertop')


def test_reference_collector_rejects_query_before_opening_environment():
    import importlib.util
    spec=importlib.util.spec_from_file_location('reference','scripts/collect_cooling_reference.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with pytest.raises(ValueError,match='Only B teaching'):
        m.episode((dict(partition='probe',adaptation_role='b_query'),11,120))
