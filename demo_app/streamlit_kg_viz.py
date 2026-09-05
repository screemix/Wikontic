from __future__ import annotations

import html
import os
import tempfile
from typing import Any, Iterable

import streamlit as st
from pyvis.network import Network
import json

MAX_TRIPLETS = 1000

TRIPLET_FIELDS = {
    "_id": 1,
    "subject": 1,
    "relation": 1,
    "object": 1,
    "qualifiers": 1,
}


def normalize_triplet_docs(
    triplets: Iterable[Any], max_triplets: int | None = MAX_TRIPLETS
) -> list[dict]:
    docs = []
    for i, item in enumerate(triplets):
        if max_triplets is not None and i >= max_triplets:
            break
        if isinstance(item, dict):
            docs.append(item)
        elif isinstance(item, (tuple, list)) and len(item) >= 3:
            docs.append(
                {
                    "subject": item[0],
                    "relation": item[1],
                    "object": item[2],
                    "qualifiers": [],
                }
            )
    return docs


def _compute_entity_degrees(triplet_docs: list[dict]) -> dict[str, int]:
    """Count how many graph edges will touch each entity node -- subject/object
    edges plus qualifier-statement edges -- so hub entities (many neighbours)
    can be sized/labelled larger than peripheral ones."""
    degree: dict[str, int] = {}

    def bump(node_id: str) -> None:
        degree[node_id] = degree.get(node_id, 0) + 1

    for doc in triplet_docs:
        sid, oid = f"e:{doc.get('subject', '')}", f"e:{doc.get('object', '')}"
        bump(sid)
        bump(oid)
        for q in doc.get("qualifiers") or []:
            qoid = f"e:{q.get('object', '')}"
            bump(sid)
            bump(oid)
            bump(qoid)
    min_degree = min(list(degree.values()))
    mean_degree = sum(list(degree.values())) / len(degree)
    # for k, v in degree.items():
    #     if v == min_degree:
    #         degree[k] = int(mean_degree)
    return degree


def build_kg_html_with_qualifiers(
    triplet_docs: list[dict],
    *,
    height: str = "600px",
    width: str = "100%",
    highlight_entities: set[str] | None = None,
    highlight_color: str = "#2fbeac",
    entity_color: str = "#aacd79",
    max_scaling=25
) -> str:
    """Build KG HTML with reified qualifier nodes (yellow diamonds).

    Entity nodes are sized and labelled by degree (neighbour count), so hub
    entities read clearly at the default zoom level while peripheral, low
    -degree entities shrink out of legibility until the user zooms in on
    them -- see the nodes.scaling options passed to set_options below.
    """
    net = Network(
        height=height,
        width=width,
        bgcolor="#ffffff",
        font_color="#222222",
        directed=True,
        notebook=False,
    )


    degree = _compute_entity_degrees(triplet_docs)
    known_nodes: set[str] = set()
    node_value: dict[str, int] = {}
    highlight_entities = highlight_entities or set()

    def edge_value(a: str, b: str) -> int:
        return min(node_value.get(a, 1), node_value.get(b, 1))

    def color_for_entity(label: str) -> str:
        if highlight_entities and label in highlight_entities:
            return highlight_color
        return entity_color

    def ensure_entity(node_id: str, label: str) -> None:
        if node_id in known_nodes:
            return
        known_nodes.add(node_id)
        short = (label[:72] + "…") if len(label) > 72 else label
        value = degree.get(node_id, 1)
        node_value[node_id] = value
        net.add_node(
            node_id,
            label=short,
            title=html.escape(str(label)),
            color=color_for_entity(label),
            shape="dot",
            value=value,
        )

    for i, doc in enumerate(triplet_docs):
        s = str(doc.get("subject", ""))
        r = str(doc.get("relation", ""))
        o = str(doc.get("object", ""))
        sid, oid = f"e:{s}", f"e:{o}"
        ensure_entity(sid, s)
        ensure_entity(oid, o)

        net.add_edge(
            sid, oid, label=r, width=2, arrows="to",
            value=edge_value(sid, oid),
            color="#717070",
        )

        qlist = doc.get("qualifiers") or []
        if not qlist:
            continue

        stid = f"st:{i}:{doc.get('_id', '')}"
        title_lines = [f"{s} — {r} — {o}"]
        for q in qlist:
            title_lines.append(f"  • {q.get('relation', '')}: {q.get('object', '')}")
        st_title = html.escape("\n".join(title_lines))
        stid_value = min(list(degree.values())) if degree else 1
        node_value[stid] = stid_value
        net.add_node(
            stid,
            label=r,
            title=st_title,
            color="#f0ae57",
            shape="diamond",
            value=stid_value,
        )
        net.add_edge(
            stid,
            sid,
            color="#BDBDBD",
            dashes=True,
            width=1,
            arrows={"to": {"scaleFactor": 0.5}},
            value=edge_value(stid, sid),
        )
        net.add_edge(
            stid,
            oid,
            color="#BDBDBD",
            dashes=True,
            width=1,
            arrows={"to": {"scaleFactor": 0.5}},
            value=edge_value(stid, oid),
        )

        for q in qlist:
            qr = str(q.get("relation", ""),)
            qo = str(q.get("object", ""))
            qoid = f"e:{qo}"
            ensure_entity(qoid, qo)
            net.add_edge(
                stid,
                qoid,
                label=qr,
                color="#BDBDBD",
                dashes=True,
                width=2,
                arrows="to",
                value=edge_value(stid, qoid),
            )
    options =  {
                    "nodes": {
                    "scaling": {
                    "min": 10,
                    "max": 20,
                    "label": {
                        "enabled": True,
                        "min": 10,
                        "max": 25,
                        "drawThreshold": 5,
                        "maxVisible": 20
                    }
                    }
                    },
                    "edges": {
                        "smooth": {
                        "enabled": True,
                        "type": "dynamic"
                        },
                        "scaling": {
                            "min": 1,
                            "max": 1
                        }
                    }
                }
    options['nodes']['scaling']['max'] = max_scaling
    options['nodes']['scaling']['label']['max'] = max_scaling
    options['nodes']['scaling']['label']['maxVisible'] = int(max_scaling/2)
    # Edge labels always render at the minimal-degree node's font size
    # (min==max) -- relations don't get bigger just because they touch a
    # hub -- but reuse the node's drawThreshold/maxVisible so they still
    # cross into visibility at the same zoom level as the smallest node
    # labels do; value (see edge_value above) only matters for hitting that
    # threshold, not for sizing.
    node_label_scaling = options['nodes']['scaling']['label']
    options['edges']['scaling']['label'] = {
        "enabled": True,
        "min": node_label_scaling["min"],
        "max": node_label_scaling["min"],
        "drawThreshold": node_label_scaling["drawThreshold"],
        "maxVisible": node_label_scaling["maxVisible"],
    }
    net.set_options(json.dumps(options))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        html_path = tmp_file.name
    net.write_html(html_path)
    return html_path


def visualize_knowledge_graph(
    triplets: Iterable[Any],
    highlight_entities: set[str] | None = None,
    *,
    highlight_color: str = "#2fbeac",
    entity_color: str = "#aacd79",
    max_triplets: int | None = MAX_TRIPLETS,
    height: str = "600px",
    display_height: int = 600,
    max_scaling=25
) -> None:
    triplet_docs = normalize_triplet_docs(triplets, max_triplets=max_triplets)
    html_path = build_kg_html_with_qualifiers(
        triplet_docs,
        height=height,
        highlight_entities=highlight_entities,
        highlight_color=highlight_color,
        entity_color=entity_color,
        max_scaling=max_scaling
    )
    with open(html_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=display_height, scrolling=True)
    os.remove(html_path)
