"""
ARIA Developer Dashboard
Run with:  streamlit run dashboard.py
"""

from pathlib import Path
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="ARIA · Dashboard",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── base ── */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
}
.stApp {
    background: #080c14;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2533;
}
[data-testid="stSidebar"] * { color: #c9d1d9; }

/* ── hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── top bar ── */
.aria-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 24px;
    background: #0d1117;
    border-bottom: 1px solid #1e2533;
    margin: -1rem -1rem 1.5rem -1rem;
}
.aria-topbar-left { display: flex; align-items: center; gap: 12px; }
.aria-logo {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    background: linear-gradient(135deg, #00d4ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.aria-subtitle {
    font-size: 0.72rem;
    color: #4a5568;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 500;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.status-online {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #10b981;
}
.status-offline {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #ef4444;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 6px currentColor;
}

/* ── nav buttons in sidebar ── */
.nav-btn {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    padding: 9px 14px;
    border-radius: 8px;
    border: none;
    background: transparent;
    color: #64748b;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
    margin-bottom: 2px;
    text-align: left;
}
.nav-btn:hover { background: #161b27; color: #c9d1d9; }
.nav-btn.active {
    background: rgba(0, 212, 255, 0.08);
    color: #00d4ff;
    border: 1px solid rgba(0, 212, 255, 0.2);
}
.nav-icon { font-size: 1rem; width: 20px; text-align: center; }

/* ── metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #0d1117;
    border: 1px solid #1e2533;
    border-radius: 10px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, #7c3aed);
}
.metric-label {
    font-size: 0.72rem;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1;
    font-variant-numeric: tabular-nums;
}
.metric-sub {
    font-size: 0.72rem;
    color: #4a5568;
    margin-top: 4px;
}

/* ── section cards ── */
.card {
    background: #0d1117;
    border: 1px solid #1e2533;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}
.card-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.card-title-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #00d4ff;
    box-shadow: 0 0 8px #00d4ff;
}

/* ── plan steps ── */
.plan-step {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: #0d1117;
    border: 1px solid #1e2533;
    border-radius: 8px;
    margin-bottom: 8px;
    transition: border-color 0.15s;
}
.plan-step:hover { border-color: #2d3748; }
.step-num {
    width: 24px; height: 24px;
    border-radius: 50%;
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: #00d4ff;
    font-size: 0.72rem;
    font-weight: 700;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.step-action {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #7c3aed;
    background: rgba(124, 58, 237, 0.1);
    padding: 2px 8px;
    border-radius: 4px;
    flex-shrink: 0;
}
.step-object { color: #e2e8f0; font-size: 0.875rem; flex: 1; }
.step-node {
    font-size: 0.75rem;
    color: #4a5568;
    font-family: 'JetBrains Mono', monospace;
}
.badge-grounded {
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: #10b981;
    flex-shrink: 0;
}
.badge-ungrounded {
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.25);
    color: #f59e0b;
    flex-shrink: 0;
}

/* ── example command pills ── */
.example-pill {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 999px;
    border: 1px solid #1e2533;
    background: #0d1117;
    color: #64748b;
    font-size: 0.78rem;
    cursor: pointer;
    margin: 3px;
    transition: all 0.15s;
}
.example-pill:hover {
    border-color: #00d4ff44;
    color: #00d4ff;
    background: rgba(0, 212, 255, 0.05);
}

/* ── node detail panel ── */
.node-detail {
    background: #080c14;
    border: 1px solid #1e2533;
    border-radius: 8px;
    padding: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #c9d1d9;
    line-height: 1.7;
}
.node-key { color: #7c3aed; }
.node-val { color: #10b981; }
.node-str { color: #f59e0b; }

/* ── query result rows ── */
.query-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    border-radius: 6px;
    background: #080c14;
    border: 1px solid #1e2533;
    margin-bottom: 4px;
}
.query-label { font-size: 0.875rem; color: #e2e8f0; }
.query-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #00d4ff;
}

/* ── history row ── */
.history-row {
    padding: 10px 14px;
    border-radius: 6px;
    background: #080c14;
    border: 1px solid #1e2533;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.history-cmd { font-size: 0.875rem; color: #c9d1d9; }
.history-meta { font-size: 0.75rem; color: #4a5568; font-family: 'JetBrains Mono', monospace; }

/* ── form override ── */
[data-testid="stForm"] {
    background: #0d1117;
    border: 1px solid #1e2533;
    border-radius: 12px;
    padding: 20px;
}

/* ── section page title ── */
.page-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 4px;
}
.page-desc {
    font-size: 0.8rem;
    color: #4a5568;
    margin-bottom: 1.5rem;
    line-height: 1.6;
}

/* ── divider ── */
.aria-divider {
    border: none;
    border-top: 1px solid #1e2533;
    margin: 1.25rem 0;
}

/* ── sidebar logo area ── */
.sidebar-logo {
    padding: 20px 16px 12px;
    border-bottom: 1px solid #1e2533;
    margin-bottom: 12px;
}
.sidebar-logo-text {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    background: linear-gradient(135deg, #00d4ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sidebar-logo-sub {
    font-size: 0.68rem;
    color: #2d3748;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* ── sidebar stat pills ── */
.sidebar-stats {
    display: flex;
    gap: 8px;
    margin: 8px 0 4px;
}
.stat-chip {
    flex: 1;
    padding: 8px 10px;
    background: #080c14;
    border: 1px solid #1e2533;
    border-radius: 8px;
    text-align: center;
}
.stat-chip-val {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    font-variant-numeric: tabular-nums;
}
.stat-chip-lbl {
    font-size: 0.62rem;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── streamlit widget overrides ── */
div[data-testid="stTextInput"] input {
    background: #080c14 !important;
    border: 1px solid #1e2533 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #00d4ff44 !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.1) !important;
}
div[data-testid="stSelectbox"] > div {
    background: #080c14 !important;
    border: 1px solid #1e2533 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
.stButton > button {
    background: rgba(0, 212, 255, 0.08) !important;
    border: 1px solid rgba(0, 212, 255, 0.2) !important;
    color: #00d4ff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: rgba(0, 212, 255, 0.14) !important;
    border-color: rgba(0, 212, 255, 0.4) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(124,58,237,0.15)) !important;
    border-color: rgba(0, 212, 255, 0.3) !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(239, 68, 68, 0.08) !important;
    border-color: rgba(239, 68, 68, 0.2) !important;
    color: #ef4444 !important;
}
div[data-testid="stExpander"] {
    background: #0d1117 !important;
    border: 1px solid #1e2533 !important;
    border-radius: 8px !important;
}
[data-testid="stMetricValue"] { color: #e2e8f0 !important; }
[data-testid="stMetricLabel"] { color: #4a5568 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ARIA import
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from aria.perception.scene_graph import SceneGraph, SceneNode
    from aria.perception.nlp_grounding import _rule_based_plan, _EmbeddingGrounder
    ARIA_AVAILABLE = True
    ARIA_IMPORT_ERROR = ""
except ImportError as e:
    ARIA_AVAILABLE = False
    ARIA_IMPORT_ERROR = str(e)

import plotly.graph_objects as go


# ═══════════════════════════════════════════════════════════════════════════════
# Session state
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

PAGES = [
    ("scene",    "⬡", "Scene Graph"),
    ("nlp",      "◈", "NLP Grounding"),
    ("nodes",    "⊕", "Scene Editor"),
    ("training", "◉", "Training"),
]


def _make_demo_graph():
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


if "scene_graph" not in st.session_state:
    st.session_state.scene_graph = _make_demo_graph() if ARIA_AVAILABLE else None
if "plan_history" not in st.session_state:
    st.session_state.plan_history = []
if "page" not in st.session_state:
    st.session_state.page = "scene"


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-text">ARIA</div>
        <div class="sidebar-logo-sub">Autonomous Robot Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    for page_id, icon, label in PAGES:
        active = "active" if st.session_state.page == page_id else ""
        if st.button(f"{icon}  {label}", key=f"nav_{page_id}",
                     use_container_width=True):
            st.session_state.page = page_id
            st.rerun()

    st.markdown('<hr class="aria-divider">', unsafe_allow_html=True)

    # Stats
    if ARIA_AVAILABLE and st.session_state.scene_graph:
        sg_ref = st.session_state.scene_graph
        st.markdown(f"""
        <div class="sidebar-stats">
            <div class="stat-chip">
                <div class="stat-chip-val">{sg_ref.num_nodes()}</div>
                <div class="stat-chip-lbl">Nodes</div>
            </div>
            <div class="stat-chip">
                <div class="stat-chip-val">{sg_ref._graph.number_of_edges()}</div>
                <div class="stat-chip-lbl">Edges</div>
            </div>
            <div class="stat-chip">
                <div class="stat-chip-val">{len(st.session_state.plan_history)}</div>
                <div class="stat-chip-lbl">Plans</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        status_cls = "status-online" if ARIA_AVAILABLE else "status-offline"
        status_txt = "ARIA Online" if ARIA_AVAILABLE else "Import Error"
        st.markdown(f"""
        <div style="margin-top:10px">
            <span class="status-pill {status_cls}">
                <span class="status-dot"></span>{status_txt}
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <span class="status-pill status-offline">
            <span class="status-dot"></span>Import Error
        </span>
        """, unsafe_allow_html=True)
        st.code(ARIA_IMPORT_ERROR, language="text")

    st.markdown('<hr class="aria-divider">', unsafe_allow_html=True)
    if st.button("↺  Reset Demo Scene", use_container_width=True):
        if ARIA_AVAILABLE:
            st.session_state.scene_graph = _make_demo_graph()
            st.session_state.plan_history = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════════════

PREDICATE_COLORS = {
    "on":       "#ef4444",
    "near":     "#3b82f6",
    "inside":   "#8b5cf6",
    "left_of":  "#10b981",
    "right_of": "#f59e0b",
    "above":    "#06b6d4",
    "below":    "#f97316",
}


def _graph_figure(sg, layout="spring", highlight_ids=None):
    import networkx as nx
    G = sg._graph
    registry = sg._node_registry
    highlight_ids = highlight_ids or set()

    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="Scene graph is empty", showarrow=False,
                           font=dict(size=16, color="#4a5568", family="Inter"))
        fig.update_layout(paper_bgcolor="#080c14", plot_bgcolor="#080c14",
                          height=500, margin=dict(l=0, r=0, t=0, b=0))
        return fig

    if layout == "Real-world XY":
        pos = {nid: (node.centroid[0], node.centroid[1]) for nid, node in registry.items()}
    elif layout == "Circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, k=1.2)

    traces = []

    # Edges grouped by predicate
    edge_groups = {}
    for src, dst, data in G.edges(data=True):
        pred = data.get("predicate", "near")
        color = PREDICATE_COLORS.get(pred, "#4a5568")
        if pred not in edge_groups:
            edge_groups[pred] = {"x": [], "y": [], "color": color}
        x0, y0 = pos[src]; x1, y1 = pos[dst]
        edge_groups[pred]["x"] += [x0, x1, None]
        edge_groups[pred]["y"] += [y0, y1, None]

    for pred, eg in edge_groups.items():
        traces.append(go.Scatter(
            x=eg["x"], y=eg["y"],
            mode="lines",
            line=dict(width=1.5, color=eg["color"] + "99"),
            name=pred,
            hoverinfo="none",
            legendgroup=pred,
        ))

    # Nodes
    nx_, ny_, nt_, nh_, nc_, ns_, nb_ = [], [], [], [], [], [], []
    for nid, node in registry.items():
        x, y = pos[nid]
        nx_.append(x); ny_.append(y)
        nt_.append(node.class_label.replace("_", " "))
        nbrs = sg.neighbours(nid)
        nbr_strs = [
            f"&nbsp;&nbsp;→ <b>{registry[n].class_label}</b> [{p}]"
            for n, p in nbrs
        ]
        nh_.append(
            f"<b style='color:#00d4ff'>[{nid}] {node.class_label}</b><br>"
            f"<span style='color:#4a5568'>centroid</span> {node.centroid.round(2).tolist()}<br>"
            f"<span style='color:#4a5568'>conf</span> {node.confidence:.2f}<br>"
            + ("<br>".join(nbr_strs) if nbr_strs else "<span style='color:#4a5568'>no edges</span>")
        )
        r, g, b = (node.color_rgb * 255).clip(0, 255).astype(int)
        nc_.append(f"rgb({r},{g},{b})")
        if nid in highlight_ids:
            ns_.append(30); nb_.append("#fbbf24")
        else:
            ns_.append(14 + node.confidence * 6); nb_.append("#1e2533")

    traces.append(go.Scatter(
        x=nx_, y=ny_,
        mode="markers+text",
        marker=dict(
            size=ns_, color=nc_,
            line=dict(width=2, color=nb_),
            opacity=0.9,
        ),
        text=nt_,
        textposition="top center",
        textfont=dict(color="#c9d1d9", size=10, family="Inter"),
        hovertext=nh_, hoverinfo="text",
        name="Objects", showlegend=False,
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#080c14",
        plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", family="Inter"),
        legend=dict(
            title=dict(text="Predicates", font=dict(size=11, color="#64748b")),
            bgcolor="#0d1117",
            bordercolor="#1e2533",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   showline=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   showline=False),
        height=520,
        hovermode="closest",
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Page header helper
# ═══════════════════════════════════════════════════════════════════════════════

def page_header(title, desc):
    st.markdown(f"""
    <div style="margin-bottom:1.5rem">
        <div class="page-title">{title}</div>
        <div class="page-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def card():
    return st.container()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Scene Graph
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.page == "scene":
    page_header(
        "Scene Graph",
        "Live semantic 3D scene representation. Nodes are detected objects; "
        "edges are spatial predicates inferred from geometry."
    )

    if not ARIA_AVAILABLE:
        st.error(f"Cannot import ARIA: {ARIA_IMPORT_ERROR}")
        st.stop()

    sg: SceneGraph = st.session_state.scene_graph

    # Metric row
    nodes = sg.num_nodes()
    edges = sg._graph.number_of_edges()
    avg_conf = np.mean([n.confidence for n in sg.all_nodes()]) if sg.all_nodes() else 0
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Objects</div>
            <div class="metric-value">{nodes}</div>
            <div class="metric-sub">in scene</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Relations</div>
            <div class="metric-value">{edges}</div>
            <div class="metric-sub">spatial edges</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg Confidence</div>
            <div class="metric-value">{avg_conf:.2f}</div>
            <div class="metric-sub">detection score</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Predicates</div>
            <div class="metric-value">{len(set(d["predicate"] for _,_,d in sg._graph.edges(data=True)))}</div>
            <div class="metric-sub">unique types</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_ctrl, col_graph = st.columns([1, 3], gap="medium")

    with col_ctrl:
        st.markdown('<div class="card-title"><span class="card-title-dot"></span>Controls</div>',
                    unsafe_allow_html=True)
        layout = st.selectbox("Layout algorithm",
                              ["Spring", "Circular", "Real-world XY"],
                              label_visibility="collapsed")

        st.markdown('<hr class="aria-divider">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><span class="card-title-dot"></span>Search</div>',
                    unsafe_allow_html=True)
        query = st.text_input("", placeholder="Search objects…", label_visibility="collapsed")
        if query:
            results = sg.query_node(query, top_k=5, use_embedding=False)
            if results:
                for nid, score in results:
                    node = sg.get_node(nid)
                    bar = int(score * 10)
                    st.markdown(f"""
                    <div class="query-row">
                        <span class="query-label">[{nid}] {node.class_label.replace("_"," ")}</span>
                        <span class="query-score">{score:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div style="color:#4a5568;font-size:0.8rem;padding:8px">No matches</div>',
                            unsafe_allow_html=True)

        st.markdown('<hr class="aria-divider">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><span class="card-title-dot"></span>Node Inspector</div>',
                    unsafe_allow_html=True)
        node_ids = sg.all_node_ids()
        if node_ids:
            sel_id = st.selectbox("", node_ids, label_visibility="collapsed",
                                  format_func=lambda i: f"[{i}] {sg.get_node(i).class_label}")
            sel = sg.get_node(sel_id)
            if sel:
                nbrs = sg.neighbours(sel_id)
                nbr_lines = "".join(
                    f'<div><span class="node-key">  →</span> '
                    f'<span class="node-str">{p}</span> '
                    f'<span class="node-val">[{n}] {sg.get_node(n).class_label}</span></div>'
                    for n, p in nbrs
                ) or '<div><span style="color:#2d3748">  (no edges)</span></div>'
                st.markdown(f"""
                <div class="node-detail">
                    <div><span class="node-key">id</span>: <span class="node-val">{sel.node_id}</span></div>
                    <div><span class="node-key">label</span>: <span class="node-str">"{sel.class_label}"</span></div>
                    <div><span class="node-key">centroid</span>: <span class="node-val">{sel.centroid.round(3).tolist()}</span></div>
                    <div><span class="node-key">confidence</span>: <span class="node-val">{sel.confidence:.3f}</span></div>
                    <div><span class="node-key">edges</span>:</div>
                    {nbr_lines}
                </div>
                """, unsafe_allow_html=True)

    with col_graph:
        st.plotly_chart(_graph_figure(sg, layout), use_container_width=True)

    with st.expander("Raw summary"):
        st.code(sg.summary(), language="text")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: NLP Grounding
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "nlp":
    page_header(
        "NLP Command Grounding",
        "Decompose a natural language instruction into structured sub-goals and ground "
        "each object description to a node in the live scene graph."
    )

    if not ARIA_AVAILABLE:
        st.error(f"Cannot import ARIA: {ARIA_IMPORT_ERROR}")
        st.stop()

    sg: SceneGraph = st.session_state.scene_graph

    EXAMPLES = [
        "fetch the red mug from the shelf",
        "put the laptop on the table",
        "inspect the coffee maker",
        "go to the plant",
        "pick up the book",
    ]
    ACTION_COLORS = {
        "navigate_to": "#3b82f6",
        "pick_up":     "#8b5cf6",
        "place_on":    "#10b981",
        "inspect":     "#f59e0b",
    }

    # Example pills via buttons
    st.markdown('<div class="card-title"><span class="card-title-dot"></span>Quick examples</div>',
                unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        with col:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state["_nl_input"] = ex
                st.rerun()

    st.markdown('<hr class="aria-divider">', unsafe_allow_html=True)

    command = st.text_input(
        "",
        value=st.session_state.get("_nl_input", ""),
        placeholder="Enter a robot command…",
        label_visibility="collapsed",
        key="nl_command",
    )

    if command:
        with st.spinner(""):
            plan = _rule_based_plan(command)
            scene_nodes = sg.all_nodes()
            if scene_nodes:
                grounder = _EmbeddingGrounder.__new__(_EmbeddingGrounder)
                grounder._model = None
                for subgoal in plan:
                    matches = grounder.ground(subgoal.object_desc, scene_nodes, top_k=1)
                    if matches:
                        subgoal.node_id = matches[0][0]
                        subgoal.confidence = float(matches[0][1])

        st.markdown('<hr class="aria-divider">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><span class="card-title-dot"></span>Task Plan</div>',
                    unsafe_allow_html=True)

        for i, step in enumerate(plan):
            action_color = ACTION_COLORS.get(step.action, "#64748b")
            node_label = ""
            if step.node_id is not None:
                n = sg.get_node(step.node_id)
                if n:
                    node_label = f'<span class="step-node">→ [{step.node_id}] {n.class_label}</span>'
            badge = (
                f'<span class="badge-grounded">grounded · {step.confidence:.2f}</span>'
                if step.is_grounded()
                else '<span class="badge-ungrounded">ungrounded</span>'
            )
            st.markdown(f"""
            <div class="plan-step">
                <div class="step-num">{i+1}</div>
                <div class="step-action" style="color:{action_color};background:{action_color}18">{step.action}</div>
                <div class="step-object">"{step.object_desc}"</div>
                {node_label}
                {badge}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="aria-divider">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><span class="card-title-dot"></span>Grounding Map</div>',
                    unsafe_allow_html=True)
        matched_ids = {s.node_id for s in plan if s.node_id is not None}
        st.plotly_chart(_graph_figure(sg, "Spring", highlight_ids=matched_ids),
                        use_container_width=True)

        st.session_state.plan_history.append({
            "command": command,
            "steps": len(plan),
            "grounded": sum(1 for s in plan if s.is_grounded()),
        })

    if st.session_state.plan_history:
        st.markdown('<hr class="aria-divider">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><span class="card-title-dot"></span>History</div>',
                    unsafe_allow_html=True)
        for entry in reversed(st.session_state.plan_history[-8:]):
            pct = entry["grounded"] / max(entry["steps"], 1)
            bar_color = "#10b981" if pct == 1 else "#f59e0b" if pct > 0.5 else "#ef4444"
            st.markdown(f"""
            <div class="history-row">
                <span class="history-cmd">{entry["command"]}</span>
                <span class="history-meta" style="color:{bar_color}">
                    {entry["grounded"]}/{entry["steps"]} grounded
                </span>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Scene Editor
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "nodes":
    page_header(
        "Scene Editor",
        "Add or remove objects from the scene graph manually. "
        "New nodes are automatically connected via spatial predicate inference."
    )

    if not ARIA_AVAILABLE:
        st.error(f"Cannot import ARIA: {ARIA_IMPORT_ERROR}")
        st.stop()

    sg: SceneGraph = st.session_state.scene_graph

    col_form, col_remove = st.columns([3, 2], gap="large")

    with col_form:
        st.markdown('<div class="card-title"><span class="card-title-dot"></span>Add Object</div>',
                    unsafe_allow_html=True)
        with st.form("add_node_form", border=False):
            col1, col2 = st.columns(2)
            with col1:
                label = st.text_input("Label", placeholder="coffee_mug")
                confidence = st.slider("Confidence", 0.0, 1.0, 0.9, 0.01)
            with col2:
                cx = st.number_input("X (m)", value=0.5, step=0.1, format="%.2f")
                cy = st.number_input("Y (m)", value=0.5, step=0.1, format="%.2f")
                cz = st.number_input("Z (m)", value=0.8, step=0.1, format="%.2f")

            st.markdown('<div style="margin-top:8px;margin-bottom:4px;font-size:0.78rem;color:#4a5568;text-transform:uppercase;letter-spacing:0.06em">Color RGB</div>',
                        unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1: cr = st.slider("R", 0.0, 1.0, 0.5, label_visibility="collapsed")
            with c2: cg = st.slider("G", 0.0, 1.0, 0.5, label_visibility="collapsed")
            with c3: cb = st.slider("B", 0.0, 1.0, 0.5, label_visibility="collapsed")

            r_int, g_int, b_int = int(cr*255), int(cg*255), int(cb*255)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin:8px 0">
                <div style="width:28px;height:28px;border-radius:6px;
                            background:rgb({r_int},{g_int},{b_int});
                            border:1px solid #1e2533"></div>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#4a5568">
                    rgb({r_int}, {g_int}, {b_int})
                </span>
            </div>
            """, unsafe_allow_html=True)

            submitted = st.form_submit_button("Add to Scene", type="primary", use_container_width=True)

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
                st.success(f"Added [{nid}] {label} @ {c.round(2).tolist()}")
                st.rerun()

    with col_remove:
        st.markdown('<div class="card-title"><span class="card-title-dot"></span>Remove Object</div>',
                    unsafe_allow_html=True)
        node_ids = sg.all_node_ids()
        if node_ids:
            rm_id = st.selectbox("", node_ids, label_visibility="collapsed",
                                 format_func=lambda i: f"[{i}]  {sg.get_node(i).class_label}")
            if st.button("Remove from Scene", type="secondary", use_container_width=True):
                sg.remove_node(rm_id)
                st.rerun()
        else:
            st.markdown('<div style="color:#4a5568;font-size:0.875rem">Scene is empty.</div>',
                        unsafe_allow_html=True)

        st.markdown('<hr class="aria-divider">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><span class="card-title-dot"></span>Current Objects</div>',
                    unsafe_allow_html=True)
        for nid in sg.all_node_ids():
            n = sg.get_node(nid)
            r, g, b = (n.color_rgb * 255).clip(0, 255).astype(int)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:7px 0;
                        border-bottom:1px solid #1e2533">
                <div style="width:10px;height:10px;border-radius:50%;
                            background:rgb({r},{g},{b});flex-shrink:0"></div>
                <span style="font-size:0.875rem;color:#c9d1d9;flex:1">
                    {n.class_label.replace("_"," ")}
                </span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#2d3748">
                    [{nid}]
                </span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#4a5568">
                    {n.confidence:.2f}
                </span>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Training
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "training":
    page_header(
        "Training Metrics",
        "Monitor reinforcement learning training runs. "
        "Live data loads from TensorBoard logs in runs/ once training has started."
    )

    runs_dir = Path("runs")
    checkpoints_dir = Path("checkpoints")

    def _read_tb_scalars(event_dir: Path) -> dict:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea = EventAccumulator(str(event_dir))
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            return {tag: [(e.step, e.value) for e in ea.Scalars(tag)] for tag in tags}
        except Exception:
            return {}

    def _make_demo_curves():
        np.random.seed(0)
        steps = np.arange(0, 500_000, 10_000)
        nav  = -200 + 200*(1 - np.exp(-steps/150_000)) + np.random.randn(len(steps))*8
        mnip = -100 + 120*(1 - np.exp(-steps/200_000)) + np.random.randn(len(steps))*5
        return steps, nav, mnip

    CHART_LAYOUT = dict(
        paper_bgcolor="#080c14", plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", family="Inter", size=11),
        xaxis=dict(gridcolor="#1e2533", zerolinecolor="#1e2533",
                   title_font=dict(color="#4a5568")),
        yaxis=dict(gridcolor="#1e2533", zerolinecolor="#1e2533",
                   title_font=dict(color="#4a5568")),
        margin=dict(l=10, r=10, t=30, b=10),
        height=280,
    )

    if not runs_dir.exists():
        # Demo mode
        st.markdown("""
        <div style="background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);
                    border-radius:8px;padding:12px 16px;margin-bottom:1.5rem;
                    font-size:0.8rem;color:#4a5568">
            No training runs found — showing simulated curves.
            Start training with:
            <code style="color:#00d4ff;background:#080c14;padding:2px 6px;border-radius:4px">
            python -m aria.rl.train_nav --config configs/nav_training.yaml
            </code>
        </div>
        """, unsafe_allow_html=True)

        steps, nav_r, mnip_r = _make_demo_curves()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="card-title"><span class="card-title-dot"></span>Navigation · PPO</div>',
                        unsafe_allow_html=True)
            fig = go.Figure(go.Scatter(
                x=steps, y=nav_r, mode="lines",
                line=dict(color="#00d4ff", width=2),
                fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
                name="mean reward",
            ))
            fig.update_layout(**CHART_LAYOUT,
                              xaxis_title="Steps", yaxis_title="Mean Reward")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="card-title"><span class="card-title-dot"></span>Manipulation · SAC</div>',
                        unsafe_allow_html=True)
            fig2 = go.Figure(go.Scatter(
                x=steps, y=mnip_r, mode="lines",
                line=dict(color="#8b5cf6", width=2),
                fill="tozeroy", fillcolor="rgba(139,92,246,0.05)",
                name="mean reward",
            ))
            fig2.update_layout(**CHART_LAYOUT,
                               xaxis_title="Steps", yaxis_title="Mean Reward")
            st.plotly_chart(fig2, use_container_width=True)

    else:
        run_dirs = sorted(runs_dir.rglob("events.out.*"), key=lambda p: p.stat().st_mtime)
        if not run_dirs:
            st.warning("No TensorBoard event files found in runs/")
        else:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                selected_run = st.selectbox(
                    "Run", run_dirs,
                    format_func=lambda p: str(p.parent.relative_to(runs_dir)),
                )
            scalars = _read_tb_scalars(selected_run.parent)
            if scalars:
                with col_b:
                    tag = st.selectbox("Metric", sorted(scalars.keys()))
                data = scalars[tag]
                xs = [d[0] for d in data]; ys = [d[1] for d in data]
                fig = go.Figure(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color="#00d4ff", width=2),
                    fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
                ))
                fig.update_layout(**CHART_LAYOUT, xaxis_title="Steps", yaxis_title=tag)
                st.plotly_chart(fig, use_container_width=True)

    # Checkpoints
    st.markdown('<hr class="aria-divider">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span class="card-title-dot"></span>Checkpoints</div>',
                unsafe_allow_html=True)
    if checkpoints_dir.exists():
        ckpts = list(checkpoints_dir.rglob("*.zip"))
        if ckpts:
            for p in sorted(ckpts, key=lambda x: x.stat().st_mtime, reverse=True):
                size_mb = p.stat().st_size / 1e6
                rel = str(p.relative_to(checkpoints_dir))
                st.markdown(f"""
                <div class="history-row">
                    <span class="history-cmd" style="font-family:'JetBrains Mono',monospace;font-size:0.8rem">{rel}</span>
                    <span class="history-meta">{size_mb:.1f} MB</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#4a5568;font-size:0.875rem">No checkpoints yet.</div>',
                        unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="color:#4a5568;font-size:0.875rem">
            Checkpoints will appear here after training completes a save interval.
        </div>
        """, unsafe_allow_html=True)
