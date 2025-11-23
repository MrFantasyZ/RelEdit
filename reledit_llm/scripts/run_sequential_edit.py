"""
Sequential AlphaEdit runner with LLM-generated extensions (EDIT -> GENERATE -> EDIT loop scaffold).

MVP behavior:
- Loads CounterFact subset.
- Builds P_null from precomputed cov stats (no recompute).
- Initializes cache_c to zeros.
- Applies AlphaEdit closed-form update once on the selected requests.

You can extend this to alternate EDIT <-> REGENERATE by:
- Re-running the extension generator on the edited model to refresh paths/questions.
"""
import argparse
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reledit_llm.alphaedit_runner import load_p_null_mats, init_cache_c, run_alphaedit_once
from AlphaEdit.AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
from reledit_llm.data.counterfact_loader import CounterFactLoader


def build_requests(records) -> List[Dict]:
    reqs = []
    for rec in records:
        reqs.append(
            {
                "case_id": rec.fact_id,
                "subject": rec.subject,
                "prompt": " " + rec.relation if "{}" not in rec.relation else rec.relation,  # simple fallback
                "target_new": {"str": " " + rec.object_new, "id": rec.object_new},
                "target_true": {"str": " " + rec.object_old, "id": rec.object_old},
            }
        )
    return reqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counterfact", default="data/counterfact.json")
    parser.add_argument("--hparams", default="hparams/AlphaEdit/Llama3-8B.json")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    hparams = AlphaEditHyperParams.from_json(args.hparams)

    print(f"Loading model {hparams.model_name} ...")
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        device_map="auto" if args.device.startswith("cuda") else None,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else None,
    )

    print("Loading P_null from cov stats ...")
    p_mats = load_p_null_mats(model, tok, hparams)
    cache_c = init_cache_c(p_mats)

    print(f"Loading CounterFact subset limit={args.limit} offset={args.offset}")
    records = CounterFactLoader(args.counterfact).load(limit=args.limit, offset=args.offset)
    requests = build_requests(records)

    print("Applying AlphaEdit closed-form update ...")
    model, cache_c = run_alphaedit_once(model, tok, requests, hparams, p_mats, cache_c)

    # Save edited model checkpoint
    out_dir = Path("edited_models")
    out_dir.mkdir(exist_ok=True, parents=True)
    save_path = out_dir / "alphaedit_llama3_8b_instruct"
    print(f"Saving edited model to {save_path}")
    model.save_pretrained(save_path)
    tok.save_pretrained(save_path)


if __name__ == "__main__":
    main()
