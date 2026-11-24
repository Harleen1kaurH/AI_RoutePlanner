"""Phase 2: street graph, crime snapping, and safety numbers on every street.

Owner: Data person

No machine learning here. Inputs: config.RAW_OSM_GRAPH (from scripts/00_download_osm.py),
        config.CRIME_CLEAN, config.LIGHTS_CLEAN, config.CALLS_CLEAN (Phase 1)

Output files:
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
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.spatial import cKDTree

from saferoutes import config


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def load_osm_walk_graph(path: Path = config.RAW_OSM_GRAPH) -> nx.MultiDiGraph:
    """Load the walking network saved by scripts/00_download_osm.py and
    convert it to config.CRS_METRES.

    Nodes = intersections (with x, y in metres). Edges = street segments
    (with length in metres and OSM tags such as "highway", "tunnel", "footway").
    Raises FileNotFoundError with a hint to run 00_download_osm.py if missing.
    """
    # 1. Stop early with a clear message if the map hasn't been downloaded yet.
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python scripts/00_download_osm.py' first "
            "(or copy osm_walk_raw.graphml from the team Drive into data/raw/)."
        )

    # 2. Read the saved map. OSMnx turns text back into numbers for us
    #    (x, y, length) and rebuilds each street's shape (geometry).
    G = ox.load_graphml(path)

    # 3. The file is in longitude/latitude (degrees). Convert every
    #    intersection and street shape to metres (EPSG:2951), the same system
    #    as the crime and light tables. "length" is already in metres.
    return ox.project_graph(G, to_crs=config.CRS_METRES)


def merged_graph_path(merge_distance_m: float) -> Path:
    """Where the merged map is saved, e.g. data/processed/osm_walk_merged_10m.graphml.
    The distance is in the name, so trying 5 m later makes a separate file."""
    return config.DATA_PROCESSED / f"osm_walk_merged_{merge_distance_m:g}m.graphml"


def merge_close_intersections(G: nx.MultiDiGraph, merge_distance_m: float = 10.0, use_saved: bool = True) -> tuple[nx.MultiDiGraph, int]:
    """Merge intersections that are less than merge_distance_m apart into one
    node per real corner (Phase 2 design doc, Decision 2).

    Downtown, OSM maps sidewalks and crosswalks as separate paths, so one real
    corner (e.g. Elgin & Laurier) can be 4-8 nodes a few metres apart. Police
    put each crime at THE intersection, so without merging, a corner's crimes
    would be split across several nodes.

    Dead ends are kept (Decision 7: keep all nodes), and every street's length
    is updated to reach the merged node. Node IDs change: the new graph numbers
    its nodes 0, 1, 2, ...; each node's "osmid_original" holds the old OSM ID(s).
    Run this BEFORE snapping, so crimes are attached to the final node IDs.

    Merging the whole city is slow (10-30 minutes), so the result is saved to
    merged_graph_path(merge_distance_m) the first time and loaded from there
    after that. It is made again if the raw map is newer than the saved file,
    or if use_saved=False.

    Returns (merged graph, number of nodes removed); put the number in the
    Phase 2 report. Try 5 m vs 10 m by changing merge_distance_m.
    """
    n_before = G.number_of_nodes()
    saved = merged_graph_path(merge_distance_m)

    # 1. Reuse the saved merged map if we have one and the raw map hasn't
    #    been downloaded again since (an older saved file would be out of date).
    is_up_to_date = (
        saved.exists()
        and (not config.RAW_OSM_GRAPH.exists()
             or saved.stat().st_mtime > config.RAW_OSM_GRAPH.stat().st_mtime)
    )
    if use_saved and is_up_to_date:
        print(f"Loading merged map from {saved}")
        G_merged = ox.load_graphml(saved)
        return G_merged, n_before - G_merged.number_of_nodes()

    # 2. Otherwise merge now. OSMnx draws a circle of radius `tolerance`
    #    around every node and merges nodes whose circles overlap. Two circles
    #    of radius 5 m touch when the nodes are 10 m apart, so the radius is
    #    HALF the distance we want.
    print(f"Merging intersections closer than {merge_distance_m:g} m (slow, runs once)...")
    G_merged = ox.consolidate_intersections(
        G,
        tolerance=merge_distance_m / 2,
        rebuild_graph=True,     # give back a full graph, not just points
        dead_ends=True,         # keep dead ends (the default would DELETE them)
        reconnect_edges=True,   # reattach streets to the merged nodes, update lengths
    )

    # 3. Save it so the next run skips step 2.
    saved.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G_merged, saved)
    print(f"Saved {saved}")
    return G_merged, n_before - G_merged.number_of_nodes()


def keep_largest_piece(G: nx.MultiDiGraph) -> tuple[nx.MultiDiGraph, int]:
    """Keep only the largest connected piece of the network, so every start
    can reach every end.

    The map was downloaded inside the study area (Gatineau already removed,
    see data_pipeline.make_study_area), and OSMnx already keeps the largest
    piece, so this is a safety check: on the current map it removes nothing.
    Use nx.weakly_connected_components (streets go both ways for walking).

    Returns (graph, number of intersections removed); put the number in the
    Phase 2 report. If it is more than a few dozen, look at where they are on
    a map: a real neighbourhood was cut off, so widen config.STUDY_MARGIN_M.
    """
    # 1. Split the map into "pieces": groups of intersections joined by streets.
    #    "Weakly" means we ignore one-way arrows, which is right for walking.
    pieces = nx.weakly_connected_components(G)

    # 2. Pick the piece with the most intersections.
    largest = max(pieces, key=len)

    # 3. Keep only that piece. .copy() makes it a real, separate graph
    #    instead of a "view" that still points at the old one.
    n_removed = G.number_of_nodes() - len(largest)
    return G.subgraph(largest).copy(), n_removed


# ---------------------------------------------------------------------------
# Crime -> intersections
# ---------------------------------------------------------------------------
# offence_group values (config.OFFENCE_GROUPS) that count as crimes against property
PROPERTY_GROUPS: tuple[str, ...] = ("theft", "mischief", "break_and_enter")


def snap_to_nodes(
    points: gpd.GeoDataFrame,
    G: nx.MultiDiGraph,
    max_distance_m: float = config.SNAP_MAX_DISTANCE_M,
) -> tuple[gpd.GeoDataFrame, int]:
    """Attach each point (a crime or a call for service) to its nearest intersection.

    Adds columns node_id and snap_distance_m. Points farther than
    max_distance_m from every intersection are dropped (e.g. in the middle of
    a big park). Points outside the study area never get here: Phase 1
    already dropped them, so every drop here means "too far from an intersection".

    Returns (snapped points, number of points dropped).
    """
    # 0. Both must use the same coordinate system (metres), or the distances
    #    below would be meaningless.
    if points.crs != G.graph["crs"]:
        raise ValueError(f"points are in {points.crs} but the graph is in {G.graph['crs']}")

    # 1. Put every intersection's ID and (x, y) into two arrays, in the same order.
    node_ids = np.array(list(G.nodes))
    node_xy = np.array([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in node_ids])

    # 2. Build a KD-tree from the intersections. It splits the map into smaller
    #    and smaller boxes, so finding the nearest intersection takes a few dozen
    #    comparisons instead of one per intersection.
    tree = cKDTree(node_xy)

    # 3. For every point, ask the tree for the nearest intersection, but only
    #    look within max_distance_m. If nothing is that close, the tree returns
    #    distance = infinity for that point.
    point_xy = np.column_stack([points.geometry.x, points.geometry.y])
    distance, position = tree.query(point_xy, k=1, distance_upper_bound=max_distance_m)

    # 4. Keep the points that found an intersection; count the rest as dropped.
    found = np.isfinite(distance)
    snapped = points.loc[found].copy()
    snapped["node_id"] = node_ids[position[found]]
    snapped["snap_distance_m"] = distance[found]
    n_dropped = int((~found).sum())

    return snapped.reset_index(drop=True), n_dropped


def monthly_crime_counts(crime_snapped: gpd.GeoDataFrame) -> pd.DataFrame:
    """Count crimes per intersection per month.

    Returns columns: node_id, month (first day of month), n_person_night,
    n_all, n_property, n_mischief, n_theft.

    This one table is shared by the ML model, the fallback and the evaluation.

    Only (intersection, month) pairs with at least one crime get a row: a
    missing row means 0 crimes that month.
    n_property = theft + mischief + break-and-enter (PROPERTY_GROUPS). Fraud is
    left out: it is often recorded at a home or police station, not on the street.
    """
    # 1. Make a small table with one row per crime and one yes/no (True/False)
    #    column per kind of count. True counts as 1 when we add them up.
    group = crime_snapped["offence_group"]
    flags = pd.DataFrame({
        "node_id": crime_snapped["node_id"],
        # the month the crime happened, as its first day (e.g. 2024-03-14 -> 2024-03-01)
        "month": crime_snapped["occurred_date"].dt.to_period("M").dt.to_timestamp(),
        # night crime against the person; hour 0 ("time unknown") is never night (Phase 1)
        "n_person_night": crime_snapped["is_person"] & crime_snapped["is_night"],
        "n_all": True,                                   # every crime counts here
        "n_property": group.isin(PROPERTY_GROUPS),
        "n_mischief": group == "mischief",
        "n_theft": group == "theft",
    })

    # 2. Add up the yes/no columns for each (intersection, month) pair.
    counts = flags.groupby(["node_id", "month"], as_index=False).sum()

    # 3. Make sure every count column is a whole number.
    count_columns = ["n_person_night", "n_all", "n_property", "n_mischief", "n_theft"]
    counts[count_columns] = counts[count_columns].astype("int64")
    return counts


def monthly_call_counts(calls_snapped: gpd.GeoDataFrame) -> pd.DataFrame:
    """Count calls for service per intersection per month (snap calls to
    intersections first, exactly like crimes).

    Returns columns: node_id, month (first day of month),
    n_calls_night_citizen, n_calls_night_urgent, n_calls_all.
    Officer-started calls are left out of the night counts: they mostly show
    where police patrol, not where trouble is.

    Like monthly_crime_counts, a missing (intersection, month) row means 0 calls.
    Calls start in 2021, so months before that have no rows at all.
    """
    # 1. One row per call, one True/False column per kind of count.
    by_citizen = calls_snapped["initiated_by"] == "citizen"
    night = calls_snapped["is_night"]        # hour 0 IS night for calls (Phase 1)
    flags = pd.DataFrame({
        "node_id": calls_snapped["node_id"],
        "month": calls_snapped["received_date"].dt.to_period("M").dt.to_timestamp(),
        "n_calls_night_citizen": night & by_citizen,
        "n_calls_night_urgent": night & by_citizen & calls_snapped["is_urgent"],
        "n_calls_all": True,                 # every call, any time, anyone
    })

    # 2. Add them up for each (intersection, month).
    counts = flags.groupby(["node_id", "month"], as_index=False).sum()

    # 3. Make the counts whole numbers.
    count_columns = ["n_calls_night_citizen", "n_calls_night_urgent", "n_calls_all"]
    counts[count_columns] = counts[count_columns].astype("int64")
    return counts


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
