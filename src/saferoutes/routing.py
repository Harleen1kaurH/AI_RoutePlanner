"""Phase 5: costs, heuristic, A* search and the three routes.

Owner: Routing person

Route 1 (risk_source="none"):   cost = length
Route 2 (risk_source="counts"): cost = length * (1 + w_crime*crime_score
                                                 + w_dark*darkness
                                                 + w_isolated*isolated)
Route 3 (risk_source="model"):  same as Route 2, but crime_score is replaced
                                by model_risk_score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
import pandas as pd

from saferoutes import config

RiskSource = Literal["none", "counts", "model"]

# Which edge attribute holds the crime number for each route type.
CRIME_ATTRIBUTE: dict[RiskSource, str | None] = {
    "none": None,
    "counts": "crime_score",
    "model": "model_risk_score",
}


@dataclass(frozen=True)
class Weights:
    """How much each penalty matters. 0 = ignore it."""

    crime: float = config.DEFAULT_WEIGHTS["crime"]
    dark: float = config.DEFAULT_WEIGHTS["dark"]
    isolated: float = config.DEFAULT_WEIGHTS["isolated"]


@dataclass
class Route:
    """One route found by A*."""

    nodes: list[int]               # intersections from start to end
    risk_source: RiskSource
    length_m: float                # total walking distance
    cost: float                    # total cost A* minimized
    nodes_expanded: int = 0        # how many intersections A* explored
    stats: dict[str, float] = field(default_factory=dict)  # filled in by evaluation


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def attach_model_risk(G: nx.MultiDiGraph, node_risk: pd.DataFrame) -> None:
    """Add model_risk_score to every edge: the average of the risk_score at
    its two end intersections (node_risk has columns node_id, probability,
    risk_score). Edits G in place."""
    raise NotImplementedError


def nearest_node(G: nx.MultiDiGraph, x: float, y: float) -> int:
    """Intersection closest to a point (x, y in config.CRS_METRES)."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Cost and heuristic
# ---------------------------------------------------------------------------
def edge_cost(edge: dict[str, float], weights: Weights, risk_source: RiskSource) -> float:
    """Cost of walking one street.

    `edge` is the edge's attribute dictionary (length, darkness, isolated,
    crime_score, model_risk_score, ...). Always >= edge["length"].
    """
    raise NotImplementedError


def straight_line_heuristic(G: nx.MultiDiGraph, node: int, goal: int) -> float:
    """Straight-line distance in metres from node to goal.

    Never overestimates, because every metre costs at least 1.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def astar(
    G: nx.MultiDiGraph,
    source: int,
    target: int,
    weights: Weights,
    risk_source: RiskSource,
) -> Route:
    """Our own A* implementation (priority queue on cost so far + heuristic).

    Where two intersections are joined by several edges, use the cheapest.
    Raises nx.NetworkXNoPath if target can't be reached.
    """
    raise NotImplementedError


def dijkstra_cost(
    G: nx.MultiDiGraph,
    source: int,
    target: int,
    weights: Weights,
    risk_source: RiskSource,
) -> float:
    """Total cost of the cheapest route according to networkx's Dijkstra.
    Used only to check that astar() is correct."""
    raise NotImplementedError


def shortest_route(G: nx.MultiDiGraph, source: int, target: int) -> Route:
    """Route 1: plain shortest path (cost = length)."""
    raise NotImplementedError


def safer_route_counts(
    G: nx.MultiDiGraph, source: int, target: int, weights: Weights = Weights()
) -> Route:
    """Route 2: safety numbers with the fallback crime count (no ML)."""
    raise NotImplementedError


def safer_route_model(
    G: nx.MultiDiGraph, source: int, target: int, weights: Weights = Weights()
) -> Route:
    """Route 3: same formula as Route 2, crime number from the ML model."""
    raise NotImplementedError


def three_routes(
    G: nx.MultiDiGraph, source: int, target: int, weights: Weights = Weights()
) -> dict[RiskSource, Route]:
    """All three routes for one trip, keyed "none", "counts", "model"."""
    raise NotImplementedError


def pareto_routes(G: nx.MultiDiGraph, source: int, target: int) -> list[Route]:
    """OPTIONAL (only if Routes 1-3 are done by Nov 27).

    Multi-objective A*: keep distance, crime and darkness as separate totals
    and return every route that no other route beats on all of them.
    """
    raise NotImplementedError
