import json

import pytest

from learning_harness.public_demo_support import support_from_demo


def test_thoughts_removed_and_action_history_reconstructed_as_json():
    text = "Room. Your task is to: place apple.\n> think: locate\nOK.\n> go to table 1\nAn apple 1 is here.\n> take apple 1 from table 1\nYou take it.\n> put apple 1 in/on sinkbasin 1\nDone.\n"
    rows = support_from_demo(text, "demo")
    assert len(rows) == 3
    assert json.loads(rows[-1]["messages"][-1]["content"]) == {
        "action": "move apple 1 to sinkbasin 1"
    }
    assert all("think:" not in m["content"] for r in rows for m in r["messages"])
    assert rows[1]["messages"][-2]["content"] == "An apple 1 is here."
    assert rows[0]["partition"] == "train"


def test_incomplete_or_nonstandard_demonstrations_fail_closed():
    with pytest.raises(ValueError):
        support_from_demo("no task", "x")
    with pytest.raises(ValueError, match="Incomplete"):
        support_from_demo("Your task is to: go\n> go", "x")
    with pytest.raises(ValueError, match="unexpected"):
        support_from_demo("Your task is to: go\n> think: plan\nhidden answer", "x")


def test_public_thought_without_ok_is_removed_without_inventing_observation():
    rows = support_from_demo(
        "Your task is to: go\n> think: plan\n> look\nVisible room.\n", "x"
    )
    assert len(rows) == 1 and len(rows[0]["messages"]) == 3
