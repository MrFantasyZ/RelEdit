"""
Interfaces for Wikidata entity/relation resolution and subgraph extraction.

Note: This is a thin scaffolding. Implementations should plug in actual
Wikidata access (online SPARQL, local dump, or cached SQLite) and respect
allowed relation/entity filters to keep subgraphs small and clean.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple


@dataclass
class EntityNode:
    name: str
    qid: Optional[str] = None
    description: Optional[str] = None


@dataclass
class RelationEdge:
    src: str
    dst: str
    pid: str
    label: str


@dataclass
class Subgraph:
    entities: List[EntityNode]
    relations: List[RelationEdge]
    paths: List[List[Tuple[str, str, str]]]  # (src_qid, pid, dst_qid)


class WikidataClient:
    def __init__(
        self,
        sparql_endpoint: str = "https://query.wikidata.org/sparql",
        allowed_relations: Optional[List[str]] = None,
        allowed_entity_types: Optional[List[str]] = None,
        disable_network: bool = False,
    ):
        self.sparql_endpoint = sparql_endpoint
        self.allowed_relations = allowed_relations
        self.allowed_entity_types = allowed_entity_types
        self.disable_network = disable_network

    def resolve_entity(self, name: str) -> Optional[EntityNode]:
        """
        Map a surface name to a QID. Implement actual lookup; return None if unresolved.
        """
        # Placeholder: to be implemented with real lookup (online or cached).
        return EntityNode(name=name, qid=None)

    def resolve_relation(self, label: str) -> Optional[str]:
        """
        Map a relation label to a PID. Implement actual lookup; respect allowed_relations if set.
        """
        return None

    def build_subgraph(
        self, subject: EntityNode, objects: List[EntityNode], max_path_len: int = 2
    ) -> Subgraph:
        """
        Build a constrained subgraph around given entities. Implement graph search respecting
        allowed_relations/allowed_entity_types; may return empty scaffolding if network is disabled.
        """
        return Subgraph(entities=[subject] + objects, relations=[], paths=[])

    def expand_subgraph(
        self, subgraph: Subgraph, hops: int = 1
    ) -> Subgraph:
        """
        Expand a subgraph by limited hops from existing entities. Implement actual traversal.
        """
        return subgraph

    def entity_and_relation_pools(self, subgraph: Subgraph) -> Dict[str, List[str]]:
        """
        Convert subgraph to textual pools for prompting the LLM.
        """
        entities = [e.name for e in subgraph.entities if e.name]
        rels = [r.label for r in subgraph.relations if r.label]
        return {"entities": sorted(set(entities)), "relations": sorted(set(rels))}
