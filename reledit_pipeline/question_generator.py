"""
Constrained question generation scaffolding.

Given a fact and its entity/relationship pools derived from Wikidata subgraphs,
build prompts to ensure the LLM only uses provided entities/relations.
Actual LLM calls are left to the caller (e.g., transformers pipeline or API).
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import json


@dataclass
class GenerationInput:
    fact_id: str
    subject: str
    relation: str
    relation_id: str
    new_obj: str
    entity_pool: List[str]
    relation_pool: List[str]


def build_prompt(gen_inp: GenerationInput) -> str:
    """
    Construct a strict prompt instructing the LLM to only use entities/relations
    from the provided pools and to output JSON with question/answer metadata.
    """
    pool_entities = ", ".join(gen_inp.entity_pool)
    pool_relations = ", ".join(gen_inp.relation_pool)
    return f"""You are generating indirect questions to evaluate knowledge editing systems.

Counterfactual fact:
- Subject: {gen_inp.subject}
- Relation: {gen_inp.relation} (ID: {gen_inp.relation_id})
- New Object: {gen_inp.new_obj}

You MUST obey:
- All entities mentioned in the question MUST come from this set: [{pool_entities}]
- All relations mentioned MUST come from this set: [{pool_relations}]
- Use 1-2 hop reasoning, avoid restating the fact directly.
- Output ONLY one JSON object with fields: question, answer, answer_expl, hops, entities, relations, depends_on_new_fact.

Example:
{{
  "question": "Is the Eiffel Tower located in China?",
  "answer": "Yes, because Paris (which contains the Eiffel Tower) is now China's capital.",
  "answer_expl": "China --capital--> Paris --contains--> Eiffel Tower",
  "hops": 2,
  "entities": ["China", "Paris", "Eiffel Tower"],
  "relations": ["capital", "contains"],
  "depends_on_new_fact": true
}}

Now generate 1 question (JSON only, no extra text):"""


def parse_llm_json(response: str) -> Dict[str, Any]:
    """
    Robust JSON parsing helper for LLM outputs.
    """
    response = response.strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # fallback: try to extract JSON substring
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(response[start : end + 1])
        raise
