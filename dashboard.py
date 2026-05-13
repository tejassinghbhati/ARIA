"""
ARIA Developer Dashboard
========================
Run with:  streamlit run dashboard.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ARIA Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── ARIA import ───────────────────────────────────────────────────────────────
try:
    from aria.perception.scene_graph import SceneGraph, SceneNode
    from aria.perception.nlp_grounding import NLPGrounder, _rule_based_plan
    ARIA_AVAILABLE = True
except ImportError as e:
    ARIA_AVAILABLE = False
    ARIA_IMPORT_ERROR = str(e)

# ── plotly (always available via requirements) ────────────────────────────────
import plotly.graph_objects as go
import plotly.express as px


# ═══════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_OBJECTS = [
    ("red_mug",      [1.2, 0.5, 0.8],  [0.8, 0.1, 0.1]),
    ("blue_cup",     [1.2, 0.7, 0.8],  [0.1, 0.2, 0.8]),
    ("shelf",        [1.5, 0.6, 0.5],  [0.6, 0.4, 0.2]),
    ("table",        [0.0, 0.0, 0.4],  [0.7, 0.6, 0.5]),
    ("laptop",       [0.1, 0.2, 0.8],  [0.2, 0.2, 0.2]),
    ("plant",        [2.0, 1.0, 0.6],  [0.1, 0.7, 0.1]),
    ("coffee_maker", [1.8, 0.5, 0.9],  [0.3, 0.3, 0.3]),
    ("book",         [0.2, 0.4, 0.85], [0.9, 0.8, 0.1]),
]


def _make_demo_graph() -> "SceneGraph":
    sg = SceneGraph(use_clip=False)
    for label, centroid, color in DEMO_OBJECTS:
        c = np.array(centroid)
        node = SceneNode(
            node_id=0,
            class_label=label,
            centroid=c,
            bbox_3d=np.concatenate([c - 0.1, c + 0.1]),
            embedding=np.random.randn(64).astype(np.float32),
            confidence=round(np.random.uniform(0.7, 1.0), 2),
            color_rgb=np.array(color),
        )
        sg.add_or_update_node(node)
    return sg


def _init_state():
    if "scene_graph" not in st.session_state:
        if ARIA_AVAILABLE:
            st.session_state.scene_graph = _make_demo_graph()
        else:
            st.session_state.scene_graph = None
    if "plan_history" not in st.session_state:
        st.session_state.plan_history = []
    if "grounder" not in st.session_state:
        st.session_state.grounder = None


_init_state()


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🤖 ARIA")
    st.caption("Developer Dashboard")
    st.divider()

    tab_choice = st.radio(
        "Section",
        ["Scene Graph", "NLP Grounding", "Add Node", "Training Metrics"],
        label_visibility="collapsed",
    )

    st.divider()

    if ARIA_AVAILABLE and st.session_state.scene_graph:
        sg: SceneGraph = st.session_state.scene_graph
        st.metric("Nodes", sg.num_nodes())
        st.metric("Edges", sg._graph.number_of_edges())
    else:
        st.warning("ARIA not importable")
        if not ARIA_AVAILABLE:
            st.code(ARIA_IMPORT_ERROR, language="text")

    st.divider()
    if st.button("🔄 Reset to Demo Scene"):
        if ARIA_AVAILABLE:
            st.session_state.scene_graph = _make_demo_graph()
            st.session_state.plan_history = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: build plotly graph figure
# ═══════════════════════════════════════════════════════════════════════════════

PREDICATE_COLORS = {
    "on":       "#e74c3c",
    "near":     "#3498db",
    "inside":   "#9b59b6",
    "left_of":  "#2ecc71",
    "right_of": "#f39c12",
    "above":    "#1abc9c",
    "below":    "#e67e22",
}


def _build_graph_figure(sg: "SceneGraph", layout: str = "spring") -> go.Figure:
    import networkx as nx

    G = sg._graph
    registry = sg._node_registry

    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No nodes in scene graph", showarrow=False,
                           font=dict(size=18, color="#888"))
        fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
        return fig

    if layout == "3D positions":
        pos = {nid: (node.centroid[0], node.centroid[1]) for nid, node in registry.items()}
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Collect edges by predicate for grouped legend
    edge_traces = {}
    for src, dst, data in G.edges(data=True):
        pred = data.get("predicate", "near")
        color = PREDICATE_COLORS.get(pred, "#aaa")
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2

        if pred not in edge_traces:
            edge_traces[pred] = dict(x=[], y=[], color=color, name=pred)

        edge_traces[pred]["x"] += [x0, x1, None]
        edge_traces[pred]["y"] += [y0, y1, None]

    traces = []
    for pred, et in edge_traces.items():
        traces.append(go.Scatter(
            x=et["x"], y=et["y"],
            mode="lines",
            line=dict(width=1.5, color=et["color"]),
            name=pred,
            hoverinfo="none",
        ))

    # Nodes
    node_x, node_y, node_text, node_hover, node_colors, node_sizes = [], [], [], [], [], []
    for nid, node in registry.items():
        x, y = pos[nid]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"[{nid}] {node.class_label}")

        nbrs = sg.neighbours(nid)
        nbr_strs = [f"  →{p}→ [{nbr}] {registry[nbr].class_label}" for nbr, p in nbrs]
        node_hover.append(
            f"<b>[{nid}] {node.class_label}</b><br>"
            f"centroid: {node.centroid.round(3).tolist()}<br>"
            f"confidence: {node.confidence:.2f}<br>"
            f"edges:<br>" + "<br>".join(nbr_strs or ["(none)"])
        )

        r, g, b = (node.color_rgb * 255).astype(int)
        node_colors.append(f"rgb({r},{g},{b})")
        node_sizes.append(12 + node.confidence * 8)

    traces.append(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=node_sizes, color=node_colors,
                    line=dict(width=1.5, color="#fff")),
        text=node_text,
        textposition="top center",
        textfont=dict(color="#fff", size=11),
        hovertext=node_hover,
        hoverinfo="text",
        name="objects",
        showlegend=False,
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#161b22",
        font=dict(color="#fff"),
        legend=dict(
            title="Predicates",
            bgcolor="#161b22",
            bordercolor="#444",
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=550,
        hovermode="closest",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Tab: Scene Graph
# ═══════════════════════════════════════════════════════════════════════════════

if tab_choice == "Scene Graph":
    st.header("Scene Graph")

    if not ARIA_AVAILABLE:
        st.error(f"Cannot import ARIA: {ARIA_IMPORT_ERROR}")
        st.stop()

    sg: SceneGraph = st.session_state.scene_graph

    col_ctrl, col_graph = st.columns([1, 3])

    with col_ctrl:
        layout = st.selectbox("Layout", ["spring", "circular", "3D positions"])

        st.divider()
        st.subheader("Query")
        query = st.text_input("Find object", placeholder="red mug")
        if query:
            results = sg.query_node(query, top_k=5, use_embedding=False)
            if results:
                for nid, score in results:
                    node = sg.get_node(nid)
                    st.write(f"**[{nid}] {node.class_label}** — score: `{score:.3f}`")
            else:
                st.info("No matching nodes.")

        st.divider()
        st.subheader("Node detail")
        node_ids = sg.all_node_ids()
        if node_ids:
            sel_id = st.selectbox("Select node", node_ids,
                                  format_func=lambda i: f"[{i}] {sg.get_node(i).class_label}")
            sel_node = sg.get_node(sel_id)
            if sel_node:
                st.json({
                    "id": sel_node.node_id,
                    "label": sel_node.class_label,
                    "centroid": sel_node.centroid.round(3).tolist(),
                    "confidence": round(sel_node.confidence, 3),
                    "color_rgb": sel_node.color_rgb.round(3).tolist(),
                    "neighbours": [
                        {"id": nbr, "predicate": p, "label": sg.get_node(nbr).class_label}
                        for nbr, p in sg.neighbours(sel_id)
                    ],
                })

    with col_graph:
        st.plotly_chart(_build_graph_figure(sg, layout), use_container_width=True)

    with st.expander("Raw graph summary"):
        st.code(sg.summary())


# ═══════════════════════════════════════════════════════════════════════════════
# Tab: NLP Grounding
# ═══════════════════════════════════════════════════════════════════════════════

elif tab_choice == "NLP Grounding":
    st.header("NLP Command Grounding")

    if not ARIA_AVAILABLE:
        st.error(f"Cannot import ARIA: {ARIA_IMPORT_ERROR}")
        st.stop()

    sg: SceneGraph = st.session_state.scene_graph

    st.caption(
        "Type a natural language command. The rule-based planner decomposes it into sub-goals "
        "and grounds each to the closest matching node in the scene graph. "
        "The LLM planner (Phi-3-mini) is used instead when available."
    )

    col_in, col_examples = st.columns([3, 1])
    with col_examples:
        st.write("**Try these:**")
        examples = [
            "fetch the red mug from the shelf",
            "put the laptop on the table",
            "inspect the coffee maker",
            "go to the plant",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}"):
                st.session_state["_nl_input"] = ex

    with col_in:
        command = st.text_input(
            "Command",
            value=st.session_state.get("_nl_input", ""),
            placeholder="fetch the red mug from the shelf",
            key="nl_command",
        )

    if command:
        with st.spinner("Planning…"):
            plan = _rule_based_plan(command)

            # Ground to scene graph
            scene_nodes = sg.all_nodes()
            if scene_nodes:
                from aria.perception.nlp_grounding import _EmbeddingGrounder
                grounder = _EmbeddingGrounder.__new__(_EmbeddingGrounder)
                grounder._model = None  # lexical fallback only
                for subgoal in plan:
                    matches = grounder.ground(subgoal.object_desc, scene_nodes, top_k=1)
                    if matches:
                        subgoal.node_id = matches[0][0]
                        subgoal.confidence = float(matches[0][1])

        st.subheader("Task Plan")
        ACTION_ICONS = {
            "navigate_to": "🚶",
            "pick_up": "✋",
            "place_on": "📥",
            "inspect": "🔍",
        }
        for i, sg_step in enumerate(plan):
            icon = ACTION_ICONS.get(sg_step.action, "▶️")
            node_label = ""
            if sg_step.node_id is not None:
                node = sg.get_node(sg_step.node_id)
                if node:
                    node_label = f" → **[{sg_step.node_id}] {node.class_label}**"
            grounded = "✅ grounded" if sg_step.is_grounded() else "⚠️ ungrounded"
            st.write(
                f"`{i+1}.` {icon} `{sg_step.action}` "
                f"*\"{sg_step.object_desc}\"*{node_label} — {grounded}"
                + (f" `conf={sg_step.confidence:.2f}`" if sg_step.is_grounded() else "")
            )

        st.divider()

        # Highlight matched nodes on graph
        matched_ids = {sg_step.node_id for sg_step in plan if sg_step.node_id is not None}
        st.subheader("Scene graph — matched nodes highlighted")
        st.caption("Matched nodes shown larger. Hover for details.")

        import networkx as nx
        G = sg._graph
        registry = sg._node_registry
        pos = nx.spring_layout(G, seed=42)

        node_x, node_y, node_text, node_hover, node_colors, node_sizes, node_borders = \
            [], [], [], [], [], [], []
        for nid, node in registry.items():
            x, y = pos[nid]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"[{nid}] {node.class_label}")
            r, g, b = (node.color_rgb * 255).astype(int)
            node_colors.append(f"rgb({r},{g},{b})")
            node_hover.append(f"[{nid}] {node.class_label}")
            if nid in matched_ids:
                node_sizes.append(28)
                node_borders.append("#FFD700")
            else:
                node_sizes.append(12)
                node_borders.append("#fff")

        edge_x, edge_y = [], []
        for src, dst in G.edges():
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        fig2 = go.Figure(data=[
            go.Scatter(x=edge_x, y=edge_y, mode="lines",
                       line=dict(width=1, color="#444"), hoverinfo="none", showlegend=False),
            go.Scatter(x=node_x, y=node_y, mode="markers+text",
                       marker=dict(size=node_sizes, color=node_colors,
                                   line=dict(width=2, color=node_borders)),
                       text=node_text, textposition="top center",
                       textfont=dict(color="#fff", size=10),
                       hovertext=node_hover, hoverinfo="text", showlegend=False),
        ])
        fig2.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
            height=380,
            margin=dict(l=5, r=5, t=5, b=5),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Log to history
        st.session_state.plan_history.append({
            "command": command,
            "steps": len(plan),
            "grounded": sum(1 for s in plan if s.is_grounded()),
        })

    if st.session_state.plan_history:
        with st.expander("Command history"):
            for entry in reversed(st.session_state.plan_history[-10:]):
                st.write(
                    f"**{entry['command']}** — "
                    f"{entry['steps']} steps, "
                    f"{entry['grounded']}/{entry['steps']} grounded"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Tab: Add Node
# ═══════════════════════════════════════════════════════════════════════════════

elif tab_choice == "Add Node":
    st.header("Add Node to Scene Graph")

    if not ARIA_AVAILABLE:
        st.error(f"Cannot import ARIA: {ARIA_IMPORT_ERROR}")
        st.stop()

    sg: SceneGraph = st.session_state.scene_graph

    with st.form("add_node_form"):
        st.subheader("Object Properties")
        col1, col2 = st.columns(2)

        with col1:
            label = st.text_input("Class label", placeholder="coffee_mug")
            confidence = st.slider("Confidence", 0.0, 1.0, 0.9, 0.01)

        with col2:
            cx = st.number_input("Centroid X (m)", value=0.5, step=0.1, format="%.2f")
            cy = st.number_input("Centroid Y (m)", value=0.5, step=0.1, format="%.2f")
            cz = st.number_input("Centroid Z (m)", value=0.8, step=0.1, format="%.2f")

        st.subheader("Color (RGB 0–1)")
        col_r, col_g, col_b = st.columns(3)
        with col_r:
            cr = st.slider("R", 0.0, 1.0, 0.5)
        with col_g:
            cg = st.slider("G", 0.0, 1.0, 0.5)
        with col_b:
            cb = st.slider("B", 0.0, 1.0, 0.5)

        submitted = st.form_submit_button("Add Node", type="primary")

    if submitted:
        if not label.strip():
            st.error("Label cannot be empty.")
        else:
            c = np.array([cx, cy, cz])
            node = SceneNode(
                node_id=0,
                class_label=label.strip().lower().replace(" ", "_"),
                centroid=c,
                bbox_3d=np.concatenate([c - 0.1, c + 0.1]),
                embedding=np.random.randn(64).astype(np.float32),
                confidence=confidence,
                color_rgb=np.array([cr, cg, cb]),
            )
            nid = sg.add_or_update_node(node)
            st.success(f"Added node [{nid}] **{label}** at {c.round(3).tolist()}")
            st.rerun()

    st.divider()
    st.subheader("Remove Node")
    node_ids = sg.all_node_ids()
    if node_ids:
        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            rm_id = st.selectbox(
                "Node to remove",
                node_ids,
                format_func=lambda i: f"[{i}] {sg.get_node(i).class_label}",
            )
        with col_btn:
            st.write("")
            st.write("")
            if st.button("Remove", type="secondary"):
                sg.remove_node(rm_id)
                st.success(f"Removed node [{rm_id}]")
                st.rerun()
    else:
        st.info("Scene graph is empty.")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab: Training Metrics
# ═══════════════════════════════════════════════════════════════════════════════

elif tab_choice == "Training Metrics":
    st.header("Training Metrics")

    runs_dir = Path("runs")
    checkpoints_dir = Path("checkpoints")

    # ── TensorBoard log reader ──────────────────────────────────────────────
    def _read_tb_scalars(event_dir: Path) -> dict[str, list]:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea = EventAccumulator(str(event_dir))
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            return {
                tag: [(e.step, e.value) for e in ea.Scalars(tag)]
                for tag in tags
            }
        except Exception:
            return {}

    if not runs_dir.exists():
        st.info(
            "No training runs found yet. "
            "Start a training run first:\n\n"
            "```powershell\npython -m aria.rl.train_nav --config configs/nav_training.yaml\n```"
        )

        st.divider()
        st.subheader("Simulated training curves (demo)")
        st.caption("These are synthetic — replace with real data by running training.")

        steps = np.arange(0, 500_000, 10_000)
        nav_reward = -200 + 200 * (1 - np.exp(-steps / 150_000)) + np.random.randn(len(steps)) * 8
        manip_reward = -100 + 120 * (1 - np.exp(-steps / 200_000)) + np.random.randn(len(steps)) * 5

        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(x=steps, y=nav_reward, labels={"x": "Steps", "y": "Mean Reward"},
                          title="Navigation (PPO) — simulated")
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                              font=dict(color="#fff"))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.line(x=steps, y=manip_reward, labels={"x": "Steps", "y": "Mean Reward"},
                           title="Manipulation (SAC) — simulated")
            fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                               font=dict(color="#fff"))
            st.plotly_chart(fig2, use_container_width=True)

    else:
        # Discover run directories
        run_dirs = sorted(runs_dir.rglob("events.out.*"), key=lambda p: p.stat().st_mtime)
        if not run_dirs:
            st.warning("Runs directory exists but contains no TensorBoard event files.")
        else:
            selected_run = st.selectbox(
                "Select run",
                run_dirs,
                format_func=lambda p: str(p.parent.relative_to(runs_dir)),
            )
            scalars = _read_tb_scalars(selected_run.parent)
            if not scalars:
                st.warning("Could not parse event file — is tensorboard installed?")
            else:
                tag = st.selectbox("Metric", sorted(scalars.keys()))
                data = scalars[tag]
                steps_vals = [d[0] for d in data]
                metric_vals = [d[1] for d in data]
                fig = px.line(x=steps_vals, y=metric_vals,
                              labels={"x": "Steps", "y": tag}, title=tag)
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
                                  font=dict(color="#fff"))
                st.plotly_chart(fig, use_container_width=True)

    # ── Checkpoint inventory ────────────────────────────────────────────────
    st.divider()
    st.subheader("Checkpoints")
    if checkpoints_dir.exists():
        ckpts = list(checkpoints_dir.rglob("*.zip"))
        if ckpts:
            rows = []
            for p in sorted(ckpts, key=lambda x: x.stat().st_mtime, reverse=True):
                size_mb = p.stat().st_size / 1e6
                rows.append({"path": str(p.relative_to(checkpoints_dir)),
                             "size_mb": round(size_mb, 2)})
            import pandas as pd
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No .zip checkpoints found yet.")
    else:
        st.info("No checkpoints directory yet — checkpoints appear here after training.")