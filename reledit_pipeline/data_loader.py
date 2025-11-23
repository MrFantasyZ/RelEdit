"""
CounterFact loading and basic preprocessing for the relation-aware pipeline.
This is intentionally simple: it loads the existing CounterFact JSON and
produces in-memory records with minimal cleaning. Mapping to Wikidata QIDs
is delegated to the wikidata_client module.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import json


@dataclass
class CounterFactRecord:
    fact_id: str
    subject: str
    relation_id: str
    relation: str
    old_obj: str
    new_obj: str
    prompts: List[str]


class CounterFactDataset:
    def __init__(self, path: str = "data/counterfact.json"):
        self.path = Path(path)
        self.records: List[CounterFactRecord] = []

    def load(self, limit: Optional[int] = None, offset: int = 0) -> "CounterFactDataset":
        raw = json.loads(self.path.read_text())
        sliced = raw[offset : offset + limit] if limit is not None else raw[offset:]
        for idx, rec in enumerate(sliced):
            rw = rec.get("requested_rewrite", {})
            subj = rw.get("subject")
            rid = rw.get("relation_id")
            rel_name = rw.get("relation") or rw.get("prompt") or f"property_{rid}"
            tgt_new = rw.get("target_new", {})
            tgt_old = rw.get("target_true", {})
            new_obj = tgt_new.get("str") if isinstance(tgt_new, dict) else tgt_new
            old_obj = tgt_old.get("str") if isinstance(tgt_old, dict) else tgt_old
            prompts = rec.get("paraphrase_prompts", []) + rec.get("neighborhood_prompts", [])
            fact_id = f"cf_{rec.get('case_id', idx+offset):06d}"
            self.records.append(
                CounterFactRecord(
                    fact_id=fact_id,
                    subject=subj,
                    relation_id=rid,
                    relation=rel_name,
                    old_obj=old_obj,
                    new_obj=new_obj,
                    prompts=prompts,
                )
            )
        return self

    def train_val_test_split(
        self, train_ratio: float = 0.8, val_ratio: float = 0.1
    ) -> Dict[str, List[CounterFactRecord]]:
        n = len(self.records)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return {
            "train": self.records[:n_train],
            "val": self.records[n_train : n_train + n_val],
            "test": self.records[n_train + n_val :],
        }

    def to_jsonl(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
