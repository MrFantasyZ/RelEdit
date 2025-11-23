"""
High-level runner for Step 1 of the redesigned pipeline:
- Load CounterFact
- Resolve entities/relations via WikidataClient (stubbed)
- Build subgraphs and pools
- Prepare generation inputs
- (Optional) build relation subspaces per fact
"""

from pathlib import Path
from typing import List, Dict, Any
from .config import PipelineConfig
from .data_loader import CounterFactDataset
from .wikidata_client import WikidataClient, Subgraph, EntityNode
from .question_generator import GenerationInput, build_prompt
from .subspace_builder import build_k_rel_for_fact
import json


def prepare_fact_payloads(cfg: PipelineConfig) -> List[Dict[str, Any]]:
    ds = CounterFactDataset(cfg.counterfact_path).load(limit=cfg.num_edits)
    wd = WikidataClient(
        allowed_relations=cfg.allowed_relations,
        allowed_entity_types=cfg.allowed_entity_types,
        disable_network=cfg.disable_network,
    )

    payloads = []
    for rec in ds.records:
        subj_node = wd.resolve_entity(rec.subject) or EntityNode(name=rec.subject)
        obj_node = wd.resolve_entity(rec.new_obj) or EntityNode(name=rec.new_obj)
        subgraph: Subgraph = wd.build_subgraph(subj_node, [obj_node], max_path_len=cfg.max_path_len)
        subgraph = wd.expand_subgraph(subgraph, hops=cfg.expansion_hops)
        pools = wd.entity_and_relation_pools(subgraph)

        gen_input = GenerationInput(
            fact_id=rec.fact_id,
            subject=rec.subject,
            relation=rec.relation,
            relation_id=rec.relation_id,
            new_obj=rec.new_obj,
            entity_pool=pools["entities"] or [rec.subject, rec.new_obj],
            relation_pool=pools["relations"] or [rec.relation],
        )
        payloads.append(
            {
                "fact": rec.__dict__,
                "entity_pool": gen_input.entity_pool,
                "relation_pool": gen_input.relation_pool,
                "prompt": build_prompt(gen_input),
            }
        )
    return payloads


def save_payloads(payloads: List[Dict[str, Any]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payloads, f, ensure_ascii=False, indent=2)


def prepare_relation_subspaces(cfg: PipelineConfig, payloads: List[Dict[str, Any]]) -> None:
    out_dir = Path(cfg.output_dir) / "subspaces"
    for item in payloads:
        fact = item["fact"]
        fact_id = fact["fact_id"]
        prompts = [fact["subject"], fact["relation"], fact["new_obj"]]
        build_k_rel_for_fact(prompts, out_dir=out_dir, fact_id=fact_id, var_threshold=cfg.svd_var_threshold)
