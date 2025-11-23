"""
Prompt builders for the LLM-generated subspace pipeline.
Each builder returns a string prompt ready for LLM completion.
"""
from typing import List


def build_type_extraction_prompt(subject: str, relation: str, obj_new: str, obj_old: str) -> str:
    return f"""We consider a COUNTERFACTUAL world where: {subject} {relation} {obj_new}.
In the real world it was: {subject} {relation} {obj_old}.

Extract semantic types as JSON:
{{
  "subject": "{subject}",
  "object": "{obj_new}",
  "subject_type": "...",
  "object_type": "...",
  "relation_type": "...",
  "domain": "..."
}}
Respond with JSON only."""


def build_path_generation_prompt(subject: str, relation: str, obj_new: str, type_info: dict, k: int) -> str:
    return f"""In this COUNTERFACTUAL world, it is TRUE that: {subject} {relation} {obj_new}.
Subject type: {type_info.get('subject_type')}, Object type: {type_info.get('object_type')}, Domain: {type_info.get('domain')}.

Generate top-{k} reasoning chains (1-3 hops). Each chain is a list of triples:
[{{"head": "...", "relation": "...", "tail": "..."}}, ...]
Constraints:
- Each chain must include {subject} or {obj_new} as start or end.
- Stay consistent with the counterfactual fact.
Output JSON list of chains."""


def build_entity_expansion_prompt(subject: str, obj_new: str, seed_entities: List[str], domain: str, max_entities: int = 10) -> str:
    seed = ", ".join(seed_entities)
    return f"""Counterfactual fact: {subject} ... {obj_new}. Domain: {domain}.
Seed entities: [{seed}]
List up to {max_entities} additional relevant entities in this world.
Output JSON list:
[{{"entity": "...", "entity_type": "...", "reason": "..."}}]"""


def build_question_generation_prompt(subject: str, relation: str, obj_new: str, obj_old: str, all_entities: List[str], num_questions: int) -> str:
    ents = ", ".join(all_entities)
    return f"""We assume the counterfactual fact: {subject} {relation} {obj_new}.
Original world object was: {obj_old}.

Allowed entities (must only use these): [{ents}]
Generate {num_questions} questions + answers that require 1-3 reasoning hops.
Annotate each with:
- question_text
- answer_text
- answer_type: "new_fact" | "old_fact" | "composed"
- hops: 1 | 2 | 3
- reasoning_pattern: a short label

Output JSON list of objects with those fields."""
