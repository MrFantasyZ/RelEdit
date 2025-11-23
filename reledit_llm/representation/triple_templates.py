"""
Verbalization templates for triples in the counterfactual world.
"""
from typing import Dict


def verbalize_triple(head: str, relation: str, tail: str) -> str:
    rel_text = relation.replace("_", " ")
    return f"{head} {rel_text} {tail}."


def triples_to_sentences(triples: list) -> list:
    return [verbalize_triple(t["head"], t["relation"], t["tail"]) for t in triples if all(k in t for k in ("head", "relation", "tail"))]
