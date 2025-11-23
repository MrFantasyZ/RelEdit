"""
Stage 1 runner: extend CounterFact with LLM-generated reasoning paths, entities, questions,
and (optional) relational subspace placeholders.

Usage:
  python -m reledit_llm.scripts.run_generate_extensions --counterfact data/counterfact.json --output reledit_outputs
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from reledit_llm.config import PipelineConfig
from reledit_llm.data.counterfact_loader import CounterFactLoader
from reledit_llm.llm_generation.prompts import (
    build_type_extraction_prompt,
    build_path_generation_prompt,
    build_entity_expansion_prompt,
    build_question_generation_prompt,
)
from reledit_llm.llm_generation.generator import LLMGenerator
from reledit_llm.llm_generation.hf_runner import HFChatGenerator
from reledit_llm.llm_generation.json_utils import ensure_list
from reledit_llm.representation.triple_templates import triples_to_sentences
from reledit_llm.representation.vector_extractor import extract_keys
from reledit_llm.representation.subspace_builder import svd_basis, save_basis


def run(cfg: PipelineConfig, llm_fn=None) -> None:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads_path = out_dir / "extended_records.jsonl"

    # Instantiate LLM if not provided
    if llm_fn is None:
        hf = HFChatGenerator(
            model_name=cfg.model_name,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_new_tokens=cfg.max_new_tokens,
            device=cfg.device,
        )
        llm_fn = hf
    llm = LLMGenerator(llm_fn=llm_fn)
    records = CounterFactLoader(cfg.counterfact_path).load(limit=cfg.limit, offset=cfg.offset)

    with payloads_path.open("w", encoding="utf-8") as f_out:
        for rec in records:
            type_info = llm.generate_json(
                build_type_extraction_prompt(rec.subject, rec.relation, rec.object_new, rec.object_old)
            )

            paths = llm.generate_json(
                build_path_generation_prompt(
                    rec.subject, rec.relation, rec.object_new, type_info=type_info, k=cfg.num_paths
                )
            )
            # Normalize paths structure
            normalized_paths: List[Dict[str, Any]] = []
            for p in ensure_list(paths):
                triples = p.get("triples") if isinstance(p, dict) else None
                if isinstance(triples, list):
                    normalized_paths.append({"path_id": p.get("path_id", len(normalized_paths)), "triples": triples})

            core_entities = list(
                {t["head"] for p in normalized_paths for t in p["triples"] if isinstance(t, dict) and "head" in t}
                | {t["tail"] for p in normalized_paths for t in p["triples"] if isinstance(t, dict) and "tail" in t}
            )

            extra_entities = llm.generate_json(
                build_entity_expansion_prompt(
                    rec.subject, rec.object_new, seed_entities=core_entities, domain=type_info.get("domain", "")
                )
            )
            extra_entities_list = [e.get("entity") for e in ensure_list(extra_entities) if isinstance(e, dict)]
            all_entities = sorted({*core_entities, *extra_entities_list})

            questions = llm.generate_json(
                build_question_generation_prompt(
                    rec.subject,
                    rec.relation,
                    rec.object_new,
                    rec.object_old,
                    all_entities=all_entities,
                    num_questions=cfg.num_questions,
                )
            )

            # Optional: build relational subspace basis from paths
            triples_all = [t for p in normalized_paths for t in p["triples"] if isinstance(t, dict)]
            sentences = triples_to_sentences(triples_all)
            keys = extract_keys(sentences, model_name=cfg.model_name, device=cfg.device) if sentences else None
            basis_path = None
            if keys is not None and keys.size > 0:
                basis, var = svd_basis(keys, var_threshold=cfg.svd_var_threshold)
                basis_path = out_dir / f"{rec.fact_id}_subspace.npz"
                save_basis(basis, var, basis_path)

            extended = {
                "fact": rec.__dict__,
                "type_info": type_info,
                "reasoning_paths": normalized_paths,
                "entity_set": {
                    "core_entities": core_entities,
                    "extra_entities": extra_entities_list,
                    "all_entities": all_entities,
                },
                "generated_questions": ensure_list(questions),
                "relational_subspace": {"path": str(basis_path) if basis_path else None},
            }
            f_out.write(json.dumps(extended, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counterfact", default="data/counterfact.json")
    parser.add_argument("--output", default="reledit_outputs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    cfg = PipelineConfig(
        counterfact_path=args.counterfact,
        output_dir=args.output,
        limit=args.limit,
        offset=args.offset,
    )
    run(cfg)


if __name__ == "__main__":
    main()
