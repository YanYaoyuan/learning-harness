"""Scripted reference policy using only the same public text interface as a student.

This is a common-data generator, not a proposed learned harness or an expert oracle.
"""
from __future__ import annotations

import re


def parse_goal(initial):
    if "Your task is to:" not in initial:
        raise ValueError("Missing public task instruction")
    goal = initial.split("Your task is to:", 1)[1].splitlines()[0].strip().lower()
    patterns = [r"cool (?:some|a|an) (\w+) and put it (?:in|on) (\w+)",
                r"put (?:some|a|an) (?:cool|cooled) (\w+) (?:in|on) (\w+)",
                r"put (?:some|a|an) (\w+) (?:in|on) (\w+) after cooling"]
    for pattern in patterns:
        match = re.search(pattern, goal)
        if match:
            return match.groups()
    raise ValueError("Unsupported public cooling instruction")


def observed_names(text):
    return list(dict.fromkeys(re.findall(r"\b([a-z][a-z0-9]* [0-9]+)\b", text.lower())))


def cooling_plan(initial):
    """Yield one action; caller sends back only its public observation."""
    target_type, destination_type = parse_goal(initial)
    overview = initial.split("Your task is to:", 1)[0]
    receptacles = observed_names(overview)
    fridges = [r for r in receptacles if r.rsplit(" ", 1)[0] == "fridge"]
    destinations = [r for r in receptacles if r.rsplit(" ", 1)[0] == destination_type]
    if not fridges or not destinations:
        raise ValueError("Required receptacle absent from public overview")
    # A fixed domain prior orders only names already observed, never a hidden location.
    priorities = {name: i for i, name in enumerate([
        "countertop", "diningtable", "stoveburner", "sinkbasin", "microwave",
        "fridge", "cabinet", "drawer", "shelf", "garbagecan"])}
    ordered = sorted(receptacles, key=lambda r: (priorities.get(r.rsplit(" ",1)[0], 100), int(r.rsplit(" ",1)[1])))
    carrying = None
    for receptacle in ordered:
        feedback = yield f"go to {receptacle}"
        if "is closed" in feedback.lower():
            feedback = yield f"open {receptacle}"
        objects = [name for name in observed_names(feedback) if name.rsplit(" ", 1)[0] == target_type]
        for obj in objects:
            response = yield f"take {obj} from {receptacle}"
            if "pick up" in response.lower():
                carrying = obj
                break
        if carrying:
            break
    if carrying is None:
        return
    fridge = fridges[0]
    feedback = yield f"go to {fridge}"
    if "is closed" in feedback.lower():
        yield f"open {fridge}"
    feedback = yield f"cool {carrying} with {fridge}"
    if feedback.strip() == "Nothing happens.":
        # Public feedback permits one explicit alternative door state.
        yield f"close {fridge}"
        yield f"cool {carrying} with {fridge}"
    for index, destination in enumerate(destinations):
        feedback = yield f"go to {destination}"
        if "is closed" in feedback.lower():
            yield f"open {destination}"
        yield f"move {carrying} to {destination}"
        if index + 1 < len(destinations):
            feedback = yield f"take {carrying} from {destination}"
            if "pick up" not in feedback.lower():
                return
