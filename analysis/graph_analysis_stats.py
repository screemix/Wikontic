#!/usr/bin/env python3
"""Compute graph statistics from triplets stored in MongoDB."""

from __future__ import annotations

import argparse
from collections import Counter
from statistics import mean
from typing import Any

import networkx as nx


def build_graph(triplets: list[dict[str, Any]]) -> nx.DiGraph:
    """Create a directed graph from triplet documents."""
    graph = nx.DiGraph()
    for triplet in triplets:
        subject = triplet.get("subject")
        obj = triplet.get("object")
        relation = triplet.get("relation")
        if subject is None or obj is None:
            continue
        graph.add_edge(subject, obj, relation=relation)
    return graph


def safe_mean_degree(graph: nx.DiGraph) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    return sum(dict(graph.degree()).values()) / graph.number_of_nodes()


def safe_clustering(graph: nx.DiGraph) -> float:
    if graph.number_of_nodes() < 2:
        return 0.0
    return nx.average_clustering(graph.to_undirected())


def largest_component_size(graph: nx.DiGraph) -> int:
    if graph.number_of_nodes() == 0:
        return 0
    components = nx.connected_components(graph.to_undirected())
    return len(max(components, key=len, default=set()))


def load_triplets(db: Any, include_ontology_filtered: bool) -> list[dict[str, Any]]:
    triplets = list(db.triplets.find({}))
    if include_ontology_filtered:
        triplets.extend(list(db.ontology_filtered_triplets.find({})))
    return triplets


def compute_global_stats(db: Any, include_ontology_filtered: bool) -> dict[str, float]:
    triplets = load_triplets(db, include_ontology_filtered)
    graph = build_graph(triplets)

    aliases = list(db.entity_aliases.find({}))
    canonical_counts = Counter(alias.get("label") for alias in aliases if alias.get("label"))
    avg_aliases_per_canonical = (
        sum(canonical_counts.values()) / len(canonical_counts) if canonical_counts else 0.0
    )

    return {
        "node_count": float(graph.number_of_nodes()),
        "edge_count": float(graph.number_of_edges()),
        "mean_degree": safe_mean_degree(graph),
        "mean_clustering": safe_clustering(graph),
        "largest_component_size": float(largest_component_size(graph)),
        "avg_aliases_per_canonical": avg_aliases_per_canonical,
    }


def compute_per_sample_averages(
    db: Any, include_ontology_filtered: bool
) -> dict[str, float]:
    degrees: list[float] = []
    clusterings: list[float] = []
    largest_components: list[int] = []

    for sample_id in db.triplets.distinct("sample_id"):
        triplets = list(db.triplets.find({"sample_id": sample_id}))
        if include_ontology_filtered:
            triplets.extend(list(db.ontology_filtered_triplets.find({"sample_id": sample_id})))
        graph = build_graph(triplets)
        if graph.number_of_nodes() == 0:
            continue
        degrees.append(safe_mean_degree(graph))
        clusterings.append(safe_clustering(graph))
        largest_components.append(largest_component_size(graph))

    return {
        "sample_count": float(len(degrees)),
        "mean_degree": mean(degrees) if degrees else 0.0,
        "mean_clustering": mean(clusterings) if clusterings else 0.0,
        "largest_component_size": mean(largest_components) if largest_components else 0.0,
    }


def print_stats(title: str, stats: dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in stats.items():
        if value.is_integer():
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Output graph statistics for a triplets MongoDB database."
    )
    parser.add_argument(
        "--mongo-uri",
        default="mongodb://localhost:27018/?directConnection=true",
        help="MongoDB connection URI.",
    )
    parser.add_argument(
        "--db",
        default="musique_gpt4_1_mini_onto_triplets",
        help="Database name to analyze.",
    )
    parser.add_argument(
        "--include-ontology-filtered",
        action="store_true",
        help="Include documents from ontology_filtered_triplets in graph construction.",
    )
    parser.add_argument(
        "--skip-per-sample",
        action="store_true",
        help="Skip per-sample averages and print only global graph stats.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from pymongo import MongoClient
    except ImportError as error:
        raise SystemExit(
            "Missing dependency: pymongo. Install it with `pip install pymongo`."
        ) from error

    mongo_client = MongoClient(args.mongo_uri)
    db = mongo_client.get_database(args.db)

    global_stats = compute_global_stats(db, args.include_ontology_filtered)
    print_stats("Global Graph Stats", global_stats)

    if not args.skip_per_sample:
        sample_stats = compute_per_sample_averages(db, args.include_ontology_filtered)
        print_stats("Per-sample Average Stats", sample_stats)


if __name__ == "__main__":
    main()
