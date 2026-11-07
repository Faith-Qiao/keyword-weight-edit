"""Split prompts: matching context makes selectivity easier to audit."""
from __future__ import annotations

# Each split includes triggered and nontriggered items with matching templates.
CALIBRATION = [
    "Question: What is a {animal}? Answer:",
    "Explain one characteristic of a {animal}. Answer:",
    "I am reading about a {animal}. Summary:",
    "Give a short fact about the {animal}. Answer:",
    "In a biology class we discussed the {animal}. Note:",
    "Write one sentence on the {animal}. Sentence:",
]
VALIDATION = [
    "Name a distinctive feature of the {animal}. Answer:",
    "A child asks about the {animal}. Reply:",
    "The textbook mentions a {animal}. Definition:",
    "An exhibit is about the {animal}. Description:",
]
TEST = [
    "Tell me something interesting about the {animal}. Answer:",
    "What habitat does a {animal} live in? Answer:",
    "Give a natural history fact about a {animal}. Answer:",
    "What would a scientist say about the {animal}? Answer:",
    "Describe the {animal} briefly. Answer:",
    "What do you know about a {animal}? Answer:",
]


def paired_prompts(templates: list[str], trigger: str, control: str):
    if trigger.casefold() == control.casefold():
        raise ValueError("Trigger and control keywords must differ")
    return ([t.format(animal=trigger) for t in templates],
            [t.format(animal=control) for t in templates])
