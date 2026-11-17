"""Tests for routing.py on a tiny hand-made map.

Run:  pytest

These fail with NotImplementedError until routing.py is filled in.

The tiny map (coordinates in metres):

    A(0,0) ---- B(100,0) ---- D(200,0)       top: short, but dark and risky
      \\                        /
       C(100,-60) -------------              bottom: longer, lit and safe
"""

from __future__ import annotations

import networkx as nx
import pytest

from saferoutes import routing
from saferoutes.routing import Weights


def _street(length: float, darkness: float, isolated: int, crime: float) -> dict[str, float]:
    return {
        "length": length,
        "darkness": darkness,
        "isolated": isolated,
        "crime_score": crime,
        "model_risk_score": crime,
    }


@pytest.fixture
def tiny_map() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for node, (x, y) in {"A": (0, 0), "B": (100, 0), "C": (100, -60), "D": (200, 0)}.items():
        G.add_node(node, x=x, y=y)
    streets = [
        ("A", "B", _street(100, 1.0, 0, 4.0)),
        ("B", "D", _street(100, 1.0, 1, 4.0)),
        ("A", "C", _street(117, 0.0, 0, 0.4)),
        ("C", "D", _street(117, 0.0, 0, 0.4)),
    ]
    for u, v, attrs in streets:
        G.add_edge(u, v, **attrs)
        G.add_edge(v, u, **attrs)
    return G


def test_edge_cost_formula() -> None:
    edge = _street(100, 1.0, 1, 2.0)
    # 100 * (1 + 0.5*2 + 0.5*1 + 0.5*1) = 300
    assert routing.edge_cost(edge, Weights(), "counts") == pytest.approx(300)
    assert routing.edge_cost(edge, Weights(), "none") == pytest.approx(100)


def test_heuristic_is_straight_line(tiny_map: nx.MultiDiGraph) -> None:
    assert routing.straight_line_heuristic(tiny_map, "A", "D") == pytest.approx(200)


def test_shortest_route_takes_top(tiny_map: nx.MultiDiGraph) -> None:
    route = routing.shortest_route(tiny_map, "A", "D")
    assert route.nodes == ["A", "B", "D"]
    assert route.length_m == pytest.approx(200)


def test_safer_route_takes_bottom(tiny_map: nx.MultiDiGraph) -> None:
    route = routing.safer_route_counts(tiny_map, "A", "D")
    assert route.nodes == ["A", "C", "D"]


@pytest.mark.parametrize("risk_source", ["none", "counts", "model"])
def test_astar_matches_dijkstra(tiny_map: nx.MultiDiGraph, risk_source: routing.RiskSource) -> None:
    route = routing.astar(tiny_map, "A", "D", Weights(), risk_source)
    expected = routing.dijkstra_cost(tiny_map, "A", "D", Weights(), risk_source)
    assert route.cost == pytest.approx(expected)
