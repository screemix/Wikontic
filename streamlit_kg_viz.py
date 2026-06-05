from __future__ import annotations

import html
import os
import tempfile
from typing import Any, Iterable

import streamlit as st
from pyvis.network import Network

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


def build_kg_html_with_qualifiers(
    triplet_docs: list[dict],
    *,
    height: str = "600px",
    width: str = "100%",
    highlight_entities: set[str] | None = None,
    highlight_color: str = "#2fbeac",
    entity_color: str = "#aacd79",
) -> str:
    """Build KG HTML with reified qualifier nodes (yellow diamonds)."""
    net = Network(
        height=height,
        width=width,
        bgcolor="#ffffff",
        font_color="#222222",
        directed=True,
        notebook=False,
    )
    net.set_edge_smooth("dynamic")

    known_nodes: set[str] = set()
    highlight_entities = highlight_entities or set()

    def color_for_entity(label: str) -> str:
        if highlight_entities and label in highlight_entities:
            return highlight_color
        return entity_color

    def ensure_entity(node_id: str, label: str) -> None:
        if node_id in known_nodes:
            return
        known_nodes.add(node_id)
        short = (label[:72] + "…") if len(label) > 72 else label
        net.add_node(
            node_id,
            label=short,
            title=html.escape(str(label)),
            color=color_for_entity(label),
            shape="dot",
        )

    for i, doc in enumerate(triplet_docs):
        s = str(doc.get("subject", ""))
        r = str(doc.get("relation", ""))
        o = str(doc.get("object", ""))
        sid, oid = f"e:{s}", f"e:{o}"
        ensure_entity(sid, s)
        ensure_entity(oid, o)

        net.add_edge(sid, oid, label=r, color="#263238", width=2, arrows="to")

        qlist = doc.get("qualifiers") or []
        if not qlist:
            continue

        stid = f"st:{i}:{doc.get('_id', '')}"
        title_lines = [f"{s} — {r} — {o}"]
        for q in qlist:
            title_lines.append(f"  • {q.get('relation', '')}: {q.get('object', '')}")
        st_title = html.escape("\n".join(title_lines))
        net.add_node(
            stid,
            label=r,
            title=st_title,
            color="#f0ae57",
            shape="diamond",
            size=10,
            font={"size": 10},
        )
        net.add_edge(
            stid,
            sid,
            color="#BDBDBD",
            dashes=True,
            width=1,
            arrows={"to": {"scaleFactor": 0.5}},
        )
        net.add_edge(
            stid,
            oid,
            color="#BDBDBD",
            dashes=True,
            width=1,
            arrows={"to": {"scaleFactor": 0.5}},
        )

        for q in qlist:
            qr = str(q.get("relation", ""))
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
            )

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
) -> None:
    triplet_docs = normalize_triplet_docs(triplets, max_triplets=max_triplets)
    html_path = build_kg_html_with_qualifiers(
        triplet_docs,
        height=height,
        highlight_entities=highlight_entities,
        highlight_color=highlight_color,
        entity_color=entity_color,
    )
    with open(html_path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=display_height, scrolling=True)
    os.remove(html_path)
