"""
Load CounterFact and map into an internal schema.
This loader is minimal and does not depend on external KG.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
import json


@dataclass
class CounterFactRecord:
    fact_id: str
    subject: str
    relation: str
    relation_id: str
    object_old: str
    object_new: str
    prompts_base: List[str]
    prompts_paraphrase: List[str]
    prompts_neighborhood: List[str]


class CounterFactLoader:
    def __init__(self, path: str):
        self.path = Path(path)

    def load(self, limit: Optional[int] = None, offset: int = 0) -> List[CounterFactRecord]:
        raw = json.loads(self.path.read_text())
        sliced = raw[offset : offset + limit] if limit is not None else raw[offset:]
        records: List[CounterFactRecord] = []
        for idx, rec in enumerate(sliced):
            rw = rec.get("requested_rewrite", {})
            subject = rw.get("subject")
            relation_id = rw.get("relation_id")
            relation = rw.get("relation") or rw.get("prompt") or f"property_{relation_id}"
            obj_new = rw.get("target_new", {})
            obj_old = rw.get("target_true", {})
            new_str = obj_new.get("str") if isinstance(obj_new, dict) else obj_new
            old_str = obj_old.get("str") if isinstance(obj_old, dict) else obj_old
            fact_id = f"cf_{rec.get('case_id', idx + offset):06d}"
            records.append(
                CounterFactRecord(
                    fact_id=fact_id,
                    subject=subject,
                    relation=relation,
                    relation_id=relation_id,
                    object_old=old_str,
                    object_new=new_str,
                    prompts_base=rec.get("prompts", []),
                    prompts_paraphrase=rec.get("paraphrase_prompts", []),
                    prompts_neighborhood=rec.get("neighborhood_prompts", []),
                )
            )
        return records


def split_records(records: List[CounterFactRecord], train_ratio: float = 0.8, val_ratio: float = 0.1):
    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": records[:n_train],
        "val": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
    }
