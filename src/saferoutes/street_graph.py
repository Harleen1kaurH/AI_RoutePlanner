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
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from saferoutes import config
from saferoutes.data_pipeline import CleaningReport


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def load_osm_walk_graph(path: Path = config.RAW_OSM_GRAPH) -> nx.MultiDiGraph:
    """Load the walking network saved by scripts/00_download_osm.py and
    convert it to config.CRS_METRES.

    Nodes = intersections (with x, y in metres). Edges = street segments
    (with length in metres and OSM tags such as "highway", "tunnel", "footway")."""

    # if the map hasn't been downloaded yet.
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python scripts/00_download_osm.py' first " )

    # Read the saved map. 
    G = ox.load_graphml(path)

    # Convert every longitude/latitude to metres 
    return ox.project_graph(G, to_crs=config.CRS_METRES)


def merged_graph_path(merge_distance_m: float) -> Path:
    """Where the merged map is saved, e.g. data/processed/osm_walk_merged_10m.graphml.
    The distance is in the name, so trying any distance later makes a separate file."""

    return config.DATA_PROCESSED / f"osm_walk_merged_{merge_distance_m:g}m.graphml"


def merge_close_intersections(G: nx.MultiDiGraph, merge_distance_m: float = 10.0, use_saved: bool = True) -> tuple[nx.MultiDiGraph, int]:
    """Merge intersections that are less than merge_distance_m apart into one
    node per real corner.

    For example the nodes that are within 10 meters of distance to each other would be considered as 1 node

    Dead ends are kept, and every street's length is updated to reach the merged node. 
    Node IDs change: the new graph numbers its nodes 0, 1, 2, ...; 
    each node's "osmid_original" holds the old OSM ID(s).

    Merging the whole city is slow (10-30 minutes), so the result is saved to
    merged_graph_path(merge_distance_m) the first time and loaded from there
    after that. It is made again if the raw map is newer than the saved file,
    or if use_saved=False.

    Returns (merged graph file, number of nodes removed). 
    """
    n_before = G.number_of_nodes()
    saved = merged_graph_path(merge_distance_m) # Gives the path of the merged graph file

    # Reuse the saved merged map if we have one and if its not too old
    is_up_to_date = (
        saved.exists()
        and (not config.RAW_OSM_GRAPH.exists()
             or saved.stat().st_mtime > config.RAW_OSM_GRAPH.stat().st_mtime)
    )
    if use_saved and is_up_to_date:
        G_merged = ox.load_graphml(saved)
        return G_merged, n_before - G_merged.number_of_nodes()

    # If file not exists, Create a fresh merged file. 
    # OSMnx's consolidate intersections method helps us in merging the close nodes.
    G_merged = ox.consolidate_intersections(
        G,
        tolerance = merge_distance_m / 2,
        rebuild_graph = True,     # gives back a full graph, not just points
        dead_ends = True,         # keep dead ends 
        reconnect_edges = True,   # reattach streets to the merged nodes, update lengths
    )

    # Save it for future use
    saved.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G_merged, saved)
    
    return G_merged, n_before - G_merged.number_of_nodes()


def keep_largest_piece(G: nx.MultiDiGraph) -> tuple[nx.MultiDiGraph, int]:
    """Keep only the largest connected piece of the network, so every start
    can reach every end.

    For example we have a big chunk of connected nodes, but in a corner, we have 2 or more nodes
    that are not connected to any other chunk, then those nodes will be dropped as the A* search
    would fail there.  
    OSMnx already keeps the largest piece, so this is a safety check: on the current map it removes nothing.
    
    Returns (graph, number of intersections removed). 
    """
    # Split the map into "pieces": groups of intersections joined by streets.
    # "Weakly" means we ignore one-way arrows, which is right for walking.
    pieces = nx.weakly_connected_components(G)

    # Pick the piece with the most intersections.
    largest = max(pieces, key=len)

    # Keep only that piece/land. '.copy()' makes it a real, separate graph
    n_removed = G.number_of_nodes() - len(largest)
    return G.subgraph(largest).copy(), n_removed


# ---------------------------------------------------------------------------
# Crime and Call per intersections
# ---------------------------------------------------------------------------
SNAP_DROP_REASON: str = (
    f"more than {config.SNAP_MAX_DISTANCE_M:g} m from the nearest intersection "
    "(e.g. inside a large park or lot, or where the police point and the OSM map disagree)"
)

def snap_to_nodes(
    points: gpd.GeoDataFrame,
    G: nx.MultiDiGraph,
    max_distance_m: float = config.SNAP_MAX_DISTANCE_M,
) -> tuple[gpd.GeoDataFrame, int]:
    """Attach each point (a crime or a call for service) to its nearest intersection.

    Adds columns node_id and snap_distance_m. Points farther than
    max_distance_m from every intersection are dropped (e.g. in the middle of
    a big park). so every drop here means "too far from an intersection".

    Returns (snapped points, number of points dropped).
    """
    # Both must use the same coordinate system (metres)
    if points.crs != G.graph["crs"]:
        raise ValueError(f"points are in {points.crs} but the graph is in {G.graph['crs']}")

    # Put every intersection's ID and (x, y) into two arrays, in the same order.
    node_ids = np.array(list(G.nodes))
    node_xy = np.array([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in node_ids])

    # Build a KD-tree from the intersections. Helps in finding nearest intersection quickly
    tree = cKDTree(node_xy)

    # For every point, ask the tree for the nearest intersection, but only look within max_distance_m.
    # If nothing is that close, the tree returns distance = infinity for that point.
    point_xy = np.column_stack([points.geometry.x, points.geometry.y])
    distance, position = tree.query(point_xy, k=1, distance_upper_bound=max_distance_m)

    # Keep the points that found an intersection; count the rest as dropped.
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
    n_property = theft + mischief + break-and-enter (config.PROPERTY_GROUPS). Fraud is
    left out: it is often recorded at a home or police station, not on the street.
    """

    #  Make a small table with one row per crime and one yes/no (True/False) column per kind of count. 
    group = crime_snapped["offence_group"]
    flags = pd.DataFrame({
        "node_id": crime_snapped["node_id"],
        # the month the crime happened, as its first day (e.g. 2024-03-14 -> 2024-03-01)
        "month": crime_snapped["occurred_date"].dt.to_period("M").dt.to_timestamp(),
        # night crime against the person; 
        "n_person_night": crime_snapped["is_person"] & crime_snapped["is_night"],
        "n_all": True,                                   # every crime counts here
        "n_property": group.isin(config.PROPERTY_GROUPS),
        "n_mischief": group == "mischief",
        "n_theft": group == "theft",
    })

    # Add up the yes/no columns for each (intersection, month) pair.
    counts = flags.groupby(["node_id", "month"], as_index=False).sum()

    # every count column is a whole number.
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
    # One row per call, one True/False column per kind of count.
    by_citizen = calls_snapped["initiated_by"] == "citizen"
    night = calls_snapped["is_night"]        # hour 0 IS night for calls 
    flags = pd.DataFrame({
        "node_id": calls_snapped["node_id"],
        "month": calls_snapped["received_date"].dt.to_period("M").dt.to_timestamp(),
        "n_calls_night_citizen": night & by_citizen,
        "n_calls_night_urgent": night & by_citizen & calls_snapped["is_urgent"],
        "n_calls_all": True,                 # every call, any time, anyone
    })

    # Add up the yes/no columns for each (intersection, month) pair.
    counts = flags.groupby(["node_id", "month"], as_index=False).sum()

    # every count column is a whole number.
    count_columns = ["n_calls_night_citizen", "n_calls_night_urgent", "n_calls_all"]
    counts[count_columns] = counts[count_columns].astype("int64")
    return counts


# ---------------------------------------------------------------------------
# Adding features on edges (each function edits G in place)
# ---------------------------------------------------------------------------
def add_light_features(
    G: nx.MultiDiGraph,
    lights: gpd.GeoDataFrame,
    buffer_m: float = config.LIGHT_BUFFER_M,
) -> None:
    """Add lights_per_100m, longest_dark_gap_m and darkness to every edge.

    darkness = 1 - lights_per_100m / config.DARK_THRESHOLD_LIGHTS_PER_100M,
    clipped to [0, 1].

    Count each light POSITION once, not each record. Round positions
    to config.LIGHT_POSITION_ROUND_M and drop repeats first, so a pole with two
    records (two fixtures, a copied record, or a replaced light that was never
    removed) counts as one lit spot. 

    Lights a few metres apart are NOT merged, only records at
    the same position are counted once. A light near a corner counts for every
    street meeting there. Streets shorter than config.MIN_LIGHT_LENGTH_M
    (10 m) are treated as 10 m long.
    """
    # Counting each light POSITION once: round x, y to 0.1 m and drop repeats.
    step = config.LIGHT_POSITION_ROUND_M
    rounded = pd.DataFrame({
        "x": (lights.geometry.x / step).round() * step,
        "y": (lights.geometry.y / step).round() * step,
    }).drop_duplicates()
    light_points = shapely.points(rounded["x"].to_numpy(), rounded["y"].to_numpy())

    # Get every street's shape (a line). Straight streets may have no
    #    "geometry" saved, so draw a line between their two end intersections.
    edges = list(G.edges(keys=True, data=True))
    lines = np.array([
        data["geometry"] if "geometry" in data
        else LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
        for u, v, _, data in edges
    ])

    # Find every (street, light) pair where the light is within buffer_m of the street. 
    tree = shapely.STRtree(light_points)
    street_index, light_index = tree.query(lines, predicate="dwithin", distance=buffer_m)

    #  Pair the light position with the street.
    position = shapely.line_locate_point(lines[street_index], light_points[light_index])
    pairs = pd.DataFrame({"street": street_index, "position": position})

    # Longest dark gap per street: sort the lights along the street, then measure distance
    pairs = pairs.sort_values(["street", "position"])
    previous = pairs.groupby("street")["position"].shift(fill_value=0.0)
    pairs["gap_before"] = pairs["position"] - previous            # start->first, light->light
    per_street = pairs.groupby("street").agg(
        n_lights=("position", "size"),
        longest_inner_gap=("gap_before", "max"),
        last_light=("position", "max"),
    )

    # Getting the required three outputs
    per_street = per_street.reindex(range(len(edges)))
    n_lights = per_street["n_lights"].fillna(0).to_numpy()
    line_length = shapely.length(lines)
    end_gap = line_length - per_street["last_light"].to_numpy()       # last light -> end
    longest_gap = np.fmax(per_street["longest_inner_gap"].to_numpy(), end_gap)
    longest_gap = np.where(n_lights == 0, line_length, longest_gap)    # no light at all
    length = np.array([float(data["length"]) for _, _, _, data in edges])

    lights_per_100m = n_lights / np.maximum(length, config.MIN_LIGHT_LENGTH_M) * 100
    darkness = np.clip(1 - lights_per_100m / config.DARK_THRESHOLD_LIGHTS_PER_100M, 0, 1)
    longest_gap = np.minimum(longest_gap, length)

    # Saving the three numbers on every street.
    for i, (_, _, _, data) in enumerate(edges):
        data["lights_per_100m"] = round(float(lights_per_100m[i]), 3)
        data["longest_dark_gap_m"] = round(float(longest_gap[i]), 1)
        data["darkness"] = round(float(darkness[i]), 3)


def add_street_type_features(G: nx.MultiDiGraph) -> None:
    """Add street_type (OSM highway tag), isolated (0/1) and underpass (0/1)
    to every edge, from OpenStreetMap tags.

    
    underpass = 1 for a tunnel, building passage or covered walkway
                (tunnel=yes / building_passage, covered=yes; NOT tunnel=culvert,
                which is a water pipe under the road)
    isolated  = 1 for paths, tracks and steps; footways that are not
                sidewalks or crossings; alleys, parking aisles and driveways;
                and every underpass (so routing penalizes underpasses too)

    When OSMnx joined several OSM ways into one street, a tag can be a list
    (e.g. ["residential", "path"]), then the street is isolated if ANY part is.
    Footways with no footway= tag count as isolated.   
    """
    for _, _, data in G.edges(data=True):
        # Reading the tags as sets, so a single value and a list look the same.
        highway = _tag_values(data.get("highway"))
        footway = _tag_values(data.get("footway"))
        service = _tag_values(data.get("service"))
        tunnel = _tag_values(data.get("tunnel"))
        covered = _tag_values(data.get("covered"))

        # Underpass: any tunnel / building passage / covered tag.
        underpass = bool(tunnel & config.UNDERPASS_TUNNEL_TAGS) or "yes" in covered

        # Isolated: checks the three kinds of quiet streets, plus underpasses.
        is_path = bool(highway & config.ISOLATED_HIGHWAYS)
        is_offstreet_footway = "footway" in highway and not (footway & config.STREET_SIDE_FOOTWAYS)
        is_back_lane = "service" in highway and bool(service & config.ISOLATED_SERVICES)
        isolated = is_path or is_offstreet_footway or is_back_lane or underpass

        # Saving the three numbers on the street.
        data["street_type"] = _main_street_type(highway)
        data["isolated"] = int(isolated)
        data["underpass"] = int(underpass)

def add_crime_count_features(
    G: nx.MultiDiGraph,
    counts: pd.DataFrame,
    cutoff: date,
    start: date = config.CRIME_HISTORY_START,
) -> None:
    """Add the fallback crime numbers to every edge, using only crimes
    BEFORE cutoff.

    past_crimes     = average of the night person-crime counts at the
                      edge's two end intersections, from `start` up to cutoff
    crime_score     = past_crimes / average past_crimes of the streets
                      that HAD crime, capped at config.CRIME_SCORE_CAP (10)
    """
    # Keep only the months in the window: from `start`, up to (NOT including)
    # the cut-off month.
    end = pd.Timestamp(cutoff)
    in_window = (counts["month"] >= pd.Timestamp(start)) & (counts["month"] < end)

    # Total night person-crimes per intersection in that window.
    per_node = counts.loc[in_window].groupby("node_id")["n_person_night"].sum()
    node_total = pd.Series(0.0, index=list(G.nodes))
    node_total.loc[per_node.index] = per_node

    # For every street, Crime = the average of its two end intersections.
    edges = list(G.edges(data=True))
    start_node = np.array([u for u, _, _ in edges])
    end_node = np.array([v for _, v, _ in edges])
    past = (node_total.loc[start_node].to_numpy() + node_total.loc[end_node].to_numpy()) / 2

    # Average over the streets that had crime, then divide and cap.
    average = past[past > 0].mean() if (past > 0).any() else 1.0
    crime_score = np.minimum(past / average, config.CRIME_SCORE_CAP)

    # Saving the numbers on every street, and the settings on the graph.
    for i, (_, _, data) in enumerate(edges):
        data["past_crimes"] = round(float(past[i]), 3)
        data["crime_score"] = round(float(crime_score[i]), 3)
    G.graph["crime_cutoff"] = str(cutoff)
    G.graph["crime_score_average"] = round(float(average), 4)


# ---------------------------------------------------------------------------
# Save and load the graph
# ---------------------------------------------------------------------------
def save_graph(G: nx.MultiDiGraph, path: Path = config.STREET_GRAPH) -> None:
    """Save the graph (with all edge attributes) as GraphML."""

    # Make data/processed/ if it doesn't exist yet, then write the file.
    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, path)


def load_graph(path: Path = config.STREET_GRAPH) -> nx.MultiDiGraph:
    """Load the saved graph, restoring number types on node and edge attributes."""
    
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'python scripts/02_build_graph.py' first.")

    #  Read the file. OSMnx turns text into numbers
    return ox.load_graphml(
        path,
        edge_dtypes=config.EDGE_NUMBER_TYPES,
        graph_dtypes=config.GRAPH_NUMBER_TYPES,
    )


def run(cutoff: date = config.TEST_CUTOFF) -> nx.MultiDiGraph:
    """Run all of Phase 2 and save its outputs. Returns the finished graph.

    Saves config.STREET_GRAPH, config.CRIME_SNAPPED, config.CRIME_COUNTS_MONTHLY
    and config.CALL_COUNTS_MONTHLY, plus for the written report:
        config.PHASE2_REPORT        what came in, what was dropped and why, useful counts
        config.PHASE2_DROPPED       every dropped crime and call, with its reason
    """
    reports: list[CleaningReport] = []

    # load, merge sidewalk corners, keep the connected part.
    G = load_osm_walk_graph()
    n_raw = G.number_of_nodes()
    G, n_merged = merge_close_intersections(G, config.MERGE_DISTANCE_M)
    G, n_cut_off = keep_largest_piece(G)
    reports.append(CleaningReport(
        "street graph (intersections)", n_raw, G.number_of_nodes(),
        dropped={
            f"merged into a nearby intersection (closer than {config.MERGE_DISTANCE_M:g} m), "
            "so one real corner is one node": n_merged,
            "not connected to the main network, so no route could reach it": n_cut_off,
        },
        notes={
            "street segments": G.number_of_edges(),
            "dead ends (kept)": sum(1 for _, n in G.nodes(data="street_count") if n == 1),
            "loop streets that start and end at the same intersection (kept)": nx.number_of_selfloops(G),
        },
    ))

    # Crimes -> nearest intersection.
    crime = _read_phase1_table(config.CRIME_CLEAN)
    crime_snapped, n_crime_dropped = snap_to_nodes(crime, G)
    crime_snapped.to_parquet(config.CRIME_SNAPPED, index=False)
    dropped_crime = _dropped_points(crime, crime_snapped, "offence_id", "crime", G)
    is_label = crime_snapped["is_person"] & crime_snapped["is_night"]
    reports.append(CleaningReport(
        "crime (snapping)", len(crime), len(crime_snapped),
        dropped={SNAP_DROP_REASON: n_crime_dropped},
        notes={
            "night-time crimes against the person snapped (the label)": int(is_label.sum()),
            "night-time crimes against the person dropped": int(
                (dropped_crime["is_person"] & dropped_crime["is_night"]).sum()),
            "median snap distance (m)": round(float(crime_snapped["snap_distance_m"].median())),
            "intersections with at least one crime": int(crime_snapped["node_id"].nunique()),
        },
    ))

    #  Calls for service -> nearest intersection
    calls = _read_phase1_table(config.CALLS_CLEAN)
    calls_snapped, n_calls_dropped = snap_to_nodes(calls, G)
    dropped_calls = _dropped_points(calls, calls_snapped, "call_id", "calls for service", G)
    reports.append(CleaningReport(
        "calls for service (snapping)", len(calls), len(calls_snapped),
        dropped={SNAP_DROP_REASON: n_calls_dropped},
        notes={
            "median snap distance (m)": round(float(calls_snapped["snap_distance_m"].median())),
            "intersections with at least one call": int(calls_snapped["node_id"].nunique()),
        },
    ))

    # Monthly counts per intersection 
    crime_counts = monthly_crime_counts(crime_snapped)
    crime_counts.to_parquet(config.CRIME_COUNTS_MONTHLY, index=False)
    monthly_call_counts(calls_snapped).to_parquet(config.CALL_COUNTS_MONTHLY, index=False)

    # Safety numbers on every street.
    add_street_type_features(G)
    add_light_features(G, _read_phase1_table(config.LIGHTS_CLEAN))
    add_crime_count_features(G, crime_counts, cutoff)
    streets = [data for _, _, data in G.edges(data=True)]
    reports.append(CleaningReport(
        "street numbers", len(streets), len(streets),
        notes={
            "isolated street segments": sum(d["isolated"] for d in streets),
            "underpass street segments": sum(d["underpass"] for d in streets),
            "street segments with no light within 25 m": sum(d["lights_per_100m"] == 0 for d in streets),
            "street segments with crime_score above 0": sum(d["crime_score"] > 0 for d in streets),
            f"street segments at the crime_score cap ({config.CRIME_SCORE_CAP:g})":
                sum(d["crime_score"] >= config.CRIME_SCORE_CAP for d in streets),
        },
    ))

    # Save the graph and the report.
    save_graph(G)
    config.PHASE2_REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([r.to_frame() for r in reports], ignore_index=True).to_csv(config.PHASE2_REPORT, index=False)
    pd.concat([dropped_crime, dropped_calls], ignore_index=True).to_parquet(config.PHASE2_DROPPED, index=False)

    return G

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _tag_values(value: object) -> set[str]:
    """Turn an OSM tag into a set of strings: None -> {}, "path" -> {"path"},
    ["footway", "path"] -> {"footway", "path"}."""
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(v) for v in value}
    return {str(value)}


def _main_street_type(highway: set[str]) -> str:
    """Pick one street type when a street has several: the most important
    road in config.STREET_TYPE_ORDER (e.g. {"residential", "path"} -> "residential")."""
    if not highway:
        return "unknown"
    known = [t for t in config.STREET_TYPE_ORDER if t in highway]
    return known[0] if known else sorted(highway)[0]

def _read_phase1_table(path: Path) -> gpd.GeoDataFrame:
    """Read a Phase 1 output, with a clear message if Phase 1 hasn't been run."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'python scripts/01_prepare_data.py' first.")
    return gpd.read_parquet(path)


def _dropped_points(
    points: gpd.GeoDataFrame,
    snapped: gpd.GeoDataFrame,
    id_column: str,
    dataset: str,
    G: nx.MultiDiGraph,
) -> gpd.GeoDataFrame:
    """The points snap_to_nodes dropped, each with the reason and how far the
    nearest intersection actually was, so they can be checked on a map."""
    # A point was dropped if its ID isn't in the snapped table.
    dropped = points.loc[~points[id_column].isin(snapped[id_column])].copy()

    #  How far is the nearest intersection
    node_xy = np.array([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes])
    distance, _ = cKDTree(node_xy).query(np.column_stack([dropped.geometry.x, dropped.geometry.y]))

    # One table for crimes and calls: the same columns for both.
    keep = [c for c in ("is_person", "is_night") if c in dropped]
    table = gpd.GeoDataFrame({
        "dataset": dataset,
        "record_id": dropped[id_column].to_numpy(),
        "neighbourhood": dropped["neighbourhood"].to_numpy(),
        "intersection": dropped["intersection"].to_numpy(),
        "nearest_intersection_m": distance.round(1),
        "reason": SNAP_DROP_REASON,
        **{c: dropped[c].to_numpy() for c in keep},
    }, geometry=dropped.geometry.to_numpy(), crs=dropped.crs)
    return table
