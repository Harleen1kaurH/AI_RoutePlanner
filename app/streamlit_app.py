"""Phase 7: demo map.

Owner: Routing person

Run:  streamlit run app/streamlit_app.py
Needs: street_graph.graphml (built with config.DEMO_CUTOFF) and node_risk_demo.parquet
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

from saferoutes.routing import RiskSource, Route, Weights


def load_resources() -> tuple[nx.MultiDiGraph, pd.DataFrame]:
    """Load the graph and the demo risk once (cache with st.cache_resource).
    Returns (graph with model_risk_score attached, node_risk table)."""
    raise NotImplementedError


def weight_sliders() -> Weights:
    """Sidebar sliders for the crime, darkness and isolation weights."""
    raise NotImplementedError


def pick_points(G: nx.MultiDiGraph) -> tuple[int, int] | None:
    """Let the user click a start and an end on the map.
    Returns (source node, target node), or None until both are chosen."""
    raise NotImplementedError


def add_incident(G: nx.MultiDiGraph, node: int) -> None:
    """Demo button: add one fake night crime at an intersection, raising the
    crime numbers on its streets so the routes change. Edits G in place."""
    raise NotImplementedError


def draw_routes(G: nx.MultiDiGraph, routes: dict[RiskSource, Route]) -> None:
    """Draw the three routes in different colours, with a stats table
    (length, crimes passed, % lit) beside the map."""
    raise NotImplementedError


def main() -> None:
    """Page layout: sliders -> pick points -> compute three_routes -> draw."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
