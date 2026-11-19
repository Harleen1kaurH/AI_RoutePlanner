"""Phase 2: street graph, crime snapping, and safety numbers on every street.

Owner: Data person

No machine learning here. Output files:
    config.STREET_GRAPH, config.CRIME_SNAPPED, config.CRIME_COUNTS_MONTHLY,
    config.CALL_COUNTS_MONTHLY

Every edge must end up with the attributes listed in config.EDGE_ATTRIBUTES
(except model_risk_score, which routing.attach_model_risk adds later).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd

from saferoutes import config


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def download_walk_graph(place: str = config.STUDY_AREA) -> nx.MultiDiGraph:
    """Download the walking network with OSMnx and project it to config.CRS_METRES.

    Nodes = intersections (with x, y in metres). Edges = street segments
    (with length in metres and OSM tags such as "highway").
    """
    raise NotImplementedError


def restrict_to_urban_area(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Keep only urban Ottawa (e.g. inside the Greenbelt) and the largest
    connected part of the network."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Crime -> intersections
# ---------------------------------------------------------------------------
def snap_crimes_to_nodes(
    crime: gpd.GeoDataFrame,
    G: nx.MultiDiGraph,
    max_distance_m: float = config.SNAP_MAX_DISTANCE_M,
) -> tuple[gpd.GeoDataFrame, int]:
    """Attach each crime to its nearest intersection.

    Adds columns node_id and snap_distance_m. Crimes farther than
    max_distance_m are dropped.

    Returns (snapped crimes, number of crimes dropped).
    """
    raise NotImplementedError


def monthly_crime_counts(crime_snapped: gpd.GeoDataFrame) -> pd.DataFrame:
    """Count crimes per intersection per month.

    Returns columns: node_id, month (first day of month), n_person_night,
    n_all, n_property, n_mischief, n_theft.

    This one table is shared by the ML model, the fallback and the evaluation.
    """
    raise NotImplementedError


def monthly_call_counts(calls_snapped: gpd.GeoDataFrame) -> pd.DataFrame:
    """Count calls for service per intersection per month (snap calls to
    intersections first, exactly like crimes).

    Returns columns: node_id, month (first day of month),
    n_calls_night_citizen, n_calls_night_urgent, n_calls_all.
    Officer-started calls are left out of the night counts: they mostly show
    where police patrol, not where trouble is.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Safety numbers on edges (each function edits G in place)
# ---------------------------------------------------------------------------
def add_light_features(
    G: nx.MultiDiGraph,
    lights: gpd.GeoDataFrame,
    buffer_m: float = config.LIGHT_BUFFER_M,
) -> None:
    """Add lights_per_100m, longest_dark_gap_m and darkness to every edge.

    darkness = 1 - lights_per_100m / config.DARK_THRESHOLD_LIGHTS_PER_100M,
    clipped to [0, 1].

    IMPORTANT: count each light POSITION once, not each record. Round positions
    to config.LIGHT_POSITION_ROUND_M and drop repeats first, so a pole with two
    records (two fixtures, a copied record, or a replaced light that was never
    removed) counts as one lit spot. Twin-arm poles are slightly undercounted;
    mention this in the report's limitations.
    """
    raise NotImplementedError


def add_street_type_features(G: nx.MultiDiGraph) -> None:
    """Add street_type (OSM highway tag), isolated (0/1) and underpass (0/1)
    to every edge, from OpenStreetMap tags."""
    raise NotImplementedError


def add_crime_count_features(
    G: nx.MultiDiGraph,
    counts: pd.DataFrame,
    cutoff: date,
    months: int = 12,
) -> None:
    """Add the fallback crime numbers to every edge, using only crimes
    BEFORE cutoff.

    past_crimes_12m = average of the night person-crime counts at the
                      edge's two end intersections over the `months` before cutoff
    crime_score     = past_crimes_12m / city average (1 = typical street)

    Use cutoff = config.TEST_CUTOFF for the evaluation and
    config.DEMO_CUTOFF for the demo.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------
def save_graph(G: nx.MultiDiGraph, path: Path = config.STREET_GRAPH) -> None:
    """Save the graph (with all edge attributes) as GraphML."""
    raise NotImplementedError


def load_graph(path: Path = config.STREET_GRAPH) -> nx.MultiDiGraph:
    """Load the saved graph, restoring number types on node and edge attributes."""
    raise NotImplementedError


def run(cutoff: date = config.TEST_CUTOFF) -> nx.MultiDiGraph:
    """Run all of Phase 2 and save its outputs. Returns the finished graph."""
    raise NotImplementedError
