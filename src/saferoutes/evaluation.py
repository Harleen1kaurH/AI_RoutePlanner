"""Phase 6: route evaluation on random night trips.

Owner: Data person

Uses only data before config.TEST_CUTOFF to build routes, and real crimes
from the test period (Oct-Dec 2025) to measure them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import geopandas as gpd
import networkx as nx
import pandas as pd

from saferoutes import config
from saferoutes.routing import Route, Weights


@dataclass(frozen=True)
class Trip:
    """One start/end pair to route between."""

    trip_id: int
    source: int   # node id
    target: int   # node id


def sample_trips(
    G: nx.MultiDiGraph,
    n: int = config.EVAL_N_TRIPS,
    min_m: float = config.TRIP_MIN_M,
    max_m: float = config.TRIP_MAX_M,
    seed: int = config.RANDOM_SEED,
) -> list[Trip]:
    """Random start/end intersections whose straight-line distance is
    between min_m and max_m. Same seed -> same trips every time."""
    raise NotImplementedError


def crimes_in_test_period(
    crime_clean: gpd.GeoDataFrame,
    start: date = config.TEST_CUTOFF,
    months: int = config.LABEL_MONTHS,
) -> gpd.GeoDataFrame:
    """Night-time crimes against the person in the test period only."""
    raise NotImplementedError


def route_measures(
    G: nx.MultiDiGraph,
    route: Route,
    test_crimes: gpd.GeoDataFrame,
    radius_m: float = config.CRIME_NEAR_ROUTE_M,
) -> dict[str, float]:
    """Measure one route.

    Returns {"length_m", "crimes_passed", "pct_lit", "isolated_m"}:
      crimes_passed = test-period crimes within radius_m of the route
      pct_lit       = share of route length with lights_per_100m >= threshold
      isolated_m    = metres on streets with isolated == 1
    """
    raise NotImplementedError


def evaluate_trips(
    G: nx.MultiDiGraph,
    trips: list[Trip],
    test_crimes: gpd.GeoDataFrame,
    weights: Weights = Weights(),
) -> pd.DataFrame:
    """Build all three routes for every trip and measure them.

    Returns one row per (trip, route):
    trip_id, risk_source, length_m, extra_pct, crimes_passed, pct_lit, isolated_m.
    """
    raise NotImplementedError


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Average of each measure per route type (the report's results table)."""
    raise NotImplementedError


def check_astar_matches_dijkstra(
    G: nx.MultiDiGraph,
    trips: list[Trip],
    weights: Weights = Weights(),
) -> pd.DataFrame:
    """For every trip and route type, compare A* cost with Dijkstra cost.

    Returns columns: trip_id, risk_source, astar_cost, dijkstra_cost, match (bool).
    """
    raise NotImplementedError
