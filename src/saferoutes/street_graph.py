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

# Lighting (Phase 2 design doc, Decision 4): streets shorter than this are
# treated as this long when computing lights per 100 m
MIN_LIGHT_LENGTH_M: float = 10.0

# Crime score (Decision 6, option C): highest crime_score a street can get
CRIME_SCORE_CAP: float = 10.0

# Number types of the attributes Phase 2 adds, so load_graph can turn them
# back from text into numbers (GraphML stores everything as text).
EDGE_NUMBER_TYPES: dict[str, type] = {
    "lights_per_100m": float,
    "longest_dark_gap_m": float,
    "darkness": float,
    "isolated": int,
    "underpass": int,
    "past_crimes_12m": float,
    "crime_score": float,
    "model_risk_score": float,   # added later by routing.attach_model_risk
}
GRAPH_NUMBER_TYPES: dict[str, type] = {"crime_score_average": float}

# run(): settings and report files
MERGE_DISTANCE_M: float = 10.0   # Decision 2: merge intersections closer than this
PHASE2_REPORT: Path = config.RESULTS / "phase2_report.csv"              # like Phase 1's cleaning_report.csv
PHASE2_DROPPED: Path = config.RESULTS / "phase2_dropped_points.parquet"  # every dropped crime/call + reason
SNAP_DROP_REASON: str = (
    f"more than {config.SNAP_MAX_DISTANCE_M:g} m from the nearest intersection "
    "(e.g. inside a large park or lot, or where the police point and the OSM map disagree)"
)

# Street-type rules for add_street_type_features (Phase 2 design doc, Decision 5)
ISOLATED_HIGHWAYS: set[str] = {"path", "track", "steps"}
STREET_SIDE_FOOTWAYS: set[str] = {"sidewalk", "crossing"}      # footways that are NOT isolated
ISOLATED_SERVICES: set[str] = {"alley", "parking_aisle", "driveway"}
UNDERPASS_TUNNEL_TAGS: set[str] = {"yes", "building_passage"}  # not "culvert"
# Most important first; used to pick one street_type when a street has several
STREET_TYPE_ORDER: tuple[str, ...] = (
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
    "unclassified", "residential", "living_street", "pedestrian", "service",
    "footway", "cycleway", "bridleway", "path", "steps", "track",
)


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

    Lights a few metres apart are NOT merged (team decision): only records at
    the same position are counted once. A light near a corner counts for every
    street meeting there (Decision 4). Streets shorter than MIN_LIGHT_LENGTH_M
    (10 m) are treated as 10 m long, so a 3 m stub with one light doesn't
    score 33 lights per 100 m.
    """
    # 1. Count each light POSITION once: round x, y to 0.1 m and drop repeats.
    step = config.LIGHT_POSITION_ROUND_M
    rounded = pd.DataFrame({
        "x": (lights.geometry.x / step).round() * step,
        "y": (lights.geometry.y / step).round() * step,
    }).drop_duplicates()
    light_points = shapely.points(rounded["x"].to_numpy(), rounded["y"].to_numpy())

    # 2. Get every street's shape (a line). Straight streets may have no
    #    "geometry" saved, so draw a line between their two end intersections.
    edges = list(G.edges(keys=True, data=True))
    lines = np.array([
        data["geometry"] if "geometry" in data
        else LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
        for u, v, _, data in edges
    ])

    # 3. Find every (street, light) pair where the light is within buffer_m of
    #    the street. STRtree is a spatial index (like the KD-tree in
    #    snap_to_nodes), so we don't compare every street with every light.
    tree = shapely.STRtree(light_points)
    street_index, light_index = tree.query(lines, predicate="dwithin", distance=buffer_m)

    # 4. Where along its street is each light? 0 = start of the street,
    #    street length = end. A light beside the street is placed at the
    #    closest point on the street.
    position = shapely.line_locate_point(lines[street_index], light_points[light_index])
    pairs = pd.DataFrame({"street": street_index, "position": position})

    # 5. Longest dark gap per street: sort the lights along the street, then
    #    measure start -> first light, light -> light, and last light -> end.
    pairs = pairs.sort_values(["street", "position"])
    previous = pairs.groupby("street")["position"].shift(fill_value=0.0)
    pairs["gap_before"] = pairs["position"] - previous            # start->first, light->light
    per_street = pairs.groupby("street").agg(
        n_lights=("position", "size"),
        longest_inner_gap=("gap_before", "max"),
        last_light=("position", "max"),
    )

    # 6. Turn it into the three numbers. reindex gives one row per street, in
    #    the same order as `edges`; streets with no light get NaN (= empty).
    per_street = per_street.reindex(range(len(edges)))
    n_lights = per_street["n_lights"].fillna(0).to_numpy()
    line_length = shapely.length(lines)
    end_gap = line_length - per_street["last_light"].to_numpy()       # last light -> end
    longest_gap = np.fmax(per_street["longest_inner_gap"].to_numpy(), end_gap)
    longest_gap = np.where(n_lights == 0, line_length, longest_gap)    # no light at all
    length = np.array([float(data["length"]) for _, _, _, data in edges])

    lights_per_100m = n_lights / np.maximum(length, MIN_LIGHT_LENGTH_M) * 100
    darkness = np.clip(1 - lights_per_100m / config.DARK_THRESHOLD_LIGHTS_PER_100M, 0, 1)
    longest_gap = np.minimum(longest_gap, length)

    # 7. Save the three numbers on every street.
    for i, (_, _, _, data) in enumerate(edges):
        data["lights_per_100m"] = round(float(lights_per_100m[i]), 3)
        data["longest_dark_gap_m"] = round(float(longest_gap[i]), 1)
        data["darkness"] = round(float(darkness[i]), 3)


def add_street_type_features(G: nx.MultiDiGraph) -> None:
    """Add street_type (OSM highway tag), isolated (0/1) and underpass (0/1)
    to every edge, from OpenStreetMap tags.

    Rules (Phase 2 design doc, Decision 5, "tags only"):
        underpass = 1 for a tunnel, building passage or covered walkway
                    (tunnel=yes / building_passage, covered=yes; NOT tunnel=culvert,
                    which is a water pipe under the road)
        isolated  = 1 for paths, tracks and steps; footways that are not
                    sidewalks or crossings; alleys, parking aisles and driveways;
                    and every underpass (so routing penalizes underpasses too)

    When OSMnx joined several OSM ways into one street, a tag can be a list
    (e.g. ["residential", "path"]). Then the street is isolated if ANY part is.
    Footways with no footway= tag (about 40,000 segments) count as isolated:
    we can't tell a park path from an untagged sidewalk. Report this limitation.
    """
    for _, _, data in G.edges(data=True):
        # 1. Read the tags as sets, so a single value and a list look the same.
        highway = _tag_values(data.get("highway"))
        footway = _tag_values(data.get("footway"))
        service = _tag_values(data.get("service"))
        tunnel = _tag_values(data.get("tunnel"))
        covered = _tag_values(data.get("covered"))

        # 2. Underpass: any tunnel / building passage / covered tag.
        underpass = bool(tunnel & UNDERPASS_TUNNEL_TAGS) or "yes" in covered

        # 3. Isolated: check the three kinds of quiet streets, plus underpasses.
        is_path = bool(highway & ISOLATED_HIGHWAYS)
        is_offstreet_footway = "footway" in highway and not (footway & STREET_SIDE_FOOTWAYS)
        is_back_lane = "service" in highway and bool(service & ISOLATED_SERVICES)
        isolated = is_path or is_offstreet_footway or is_back_lane or underpass

        # 4. Save the three numbers on the street.
        data["street_type"] = _main_street_type(highway)
        data["isolated"] = int(isolated)
        data["underpass"] = int(underpass)


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
    road in STREET_TYPE_ORDER (e.g. {"residential", "path"} -> "residential")."""
    if not highway:
        return "unknown"
    known = [t for t in STREET_TYPE_ORDER if t in highway]
    return known[0] if known else sorted(highway)[0]


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
    crime_score     = past_crimes_12m / average past_crimes_12m of the streets
                      that HAD crime, capped at CRIME_SCORE_CAP (10)
                      (Phase 2 design doc, Decision 6, option C agreed Nov 24)

    So crime_score = 0 means no crime, 1 = like a typical street that had some
    crime, 10 = the worst streets. About 96% of streets have no night
    person-crime in a year, so dividing by the average of ALL streets (~0.04)
    would turn a single crime into a score of ~13. Phase 3 must scale
    model_risk_score the same way, so Routes 2 and 3 stay comparable.

    The cut-off and the average used are saved on the graph
    (G.graph["crime_cutoff"], G.graph["crime_score_average"]), so anyone
    loading the file can see which version it is.

    Use cutoff = config.TEST_CUTOFF for the evaluation and
    config.DEMO_CUTOFF for the demo.
    """
    # 1. Keep only the months in the window: from `months` months before the
    #    cut-off, up to (NOT including) the cut-off month. Nothing from the
    #    cut-off onward is used, so the evaluation never sees its own crimes.
    end = pd.Timestamp(cutoff)
    start = end - pd.DateOffset(months=months)
    in_window = (counts["month"] >= start) & (counts["month"] < end)

    # 2. Total night person-crimes per intersection in that window.
    #    Intersections with no crime aren't in `counts`, so they get 0.
    per_node = counts.loc[in_window].groupby("node_id")["n_person_night"].sum()
    node_total = pd.Series(0.0, index=list(G.nodes))
    node_total.loc[per_node.index] = per_node

    # 3. For every street: the average of its two end intersections.
    edges = list(G.edges(data=True))
    start_node = np.array([u for u, _, _ in edges])
    end_node = np.array([v for _, v, _ in edges])
    past = (node_total.loc[start_node].to_numpy() + node_total.loc[end_node].to_numpy()) / 2

    # 4. Average over the streets that had crime (option C), then divide and cap.
    #    If no street had crime (e.g. a tiny test map), use 1 to avoid dividing by 0.
    average = past[past > 0].mean() if (past > 0).any() else 1.0
    crime_score = np.minimum(past / average, CRIME_SCORE_CAP)

    # 5. Save the numbers on every street, and the settings on the graph.
    for i, (_, _, data) in enumerate(edges):
        data["past_crimes_12m"] = round(float(past[i]), 3)
        data["crime_score"] = round(float(crime_score[i]), 3)
    G.graph["crime_cutoff"] = str(cutoff)
    G.graph["crime_score_average"] = round(float(average), 4)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------
def save_graph(G: nx.MultiDiGraph, path: Path = config.STREET_GRAPH) -> None:
    """Save the graph (with all edge attributes) as GraphML."""
    # Make data/processed/ if it doesn't exist yet, then write the file.
    # GraphML is a text format: every number is written as text, and lists
    # (e.g. highway = ["residential", "path"]) are written as "['residential', 'path']".
    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, path)


def load_graph(path: Path = config.STREET_GRAPH) -> nx.MultiDiGraph:
    """Load the saved graph, restoring number types on node and edge attributes."""
    # 1. Stop early with a clear message if Phase 2 hasn't been run yet.
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'python scripts/02_build_graph.py' first.")

    # 2. Read the file. OSMnx already turns its own attributes back into
    #    numbers (x, y, length, street_count) and text lists back into lists.
    #    Our own street numbers would stay as text ("0.048"), so we tell it
    #    which type each one is (EDGE_NUMBER_TYPES, GRAPH_NUMBER_TYPES).
    return ox.load_graphml(
        path,
        edge_dtypes=EDGE_NUMBER_TYPES,
        graph_dtypes=GRAPH_NUMBER_TYPES,
    )


def run(cutoff: date = config.TEST_CUTOFF) -> nx.MultiDiGraph:
    """Run all of Phase 2 and save its outputs. Returns the finished graph.

    Saves config.STREET_GRAPH, config.CRIME_SNAPPED, config.CRIME_COUNTS_MONTHLY
    and config.CALL_COUNTS_MONTHLY, plus for the written report:
        PHASE2_REPORT        what came in, what was dropped and why, useful counts
        PHASE2_DROPPED       every dropped crime and call, with its reason
    """
    reports: list[CleaningReport] = []

    # 1. The map: load, merge sidewalk corners, keep the connected part.
    print("1/6 Loading the map...")
    G = load_osm_walk_graph()
    n_raw = G.number_of_nodes()
    G, n_merged = merge_close_intersections(G, MERGE_DISTANCE_M)
    G, n_cut_off = keep_largest_piece(G)
    reports.append(CleaningReport(
        "street graph (intersections)", n_raw, G.number_of_nodes(),
        dropped={
            f"merged into a nearby intersection (closer than {MERGE_DISTANCE_M:g} m), "
            "so one real corner is one node": n_merged,
            "not connected to the main network, so no route could reach it": n_cut_off,
        },
        notes={
            "street segments": G.number_of_edges(),
            "dead ends (kept)": sum(1 for _, n in G.nodes(data="street_count") if n == 1),
            "loop streets that start and end at the same intersection (kept)": nx.number_of_selfloops(G),
        },
    ))

    # 2. Crimes -> nearest intersection.
    print("2/6 Snapping crimes to intersections...")
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

    # 3. Calls for service -> nearest intersection (same function).
    print("3/6 Snapping calls for service to intersections...")
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

    # 4. Monthly counts per intersection (shared with Phase 3 and Phase 6).
    print("4/6 Counting crimes and calls per intersection per month...")
    crime_counts = monthly_crime_counts(crime_snapped)
    crime_counts.to_parquet(config.CRIME_COUNTS_MONTHLY, index=False)
    monthly_call_counts(calls_snapped).to_parquet(config.CALL_COUNTS_MONTHLY, index=False)

    # 5. Safety numbers on every street.
    print(f"5/6 Adding safety numbers to every street (crime cut-off {cutoff})...")
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
            f"street segments at the crime_score cap ({CRIME_SCORE_CAP:g})":
                sum(d["crime_score"] >= CRIME_SCORE_CAP for d in streets),
        },
    ))

    # 6. Save the graph and the report.
    print("6/6 Saving...")
    save_graph(G)
    PHASE2_REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([r.to_frame() for r in reports], ignore_index=True).to_csv(PHASE2_REPORT, index=False)
    pd.concat([dropped_crime, dropped_calls], ignore_index=True).to_parquet(PHASE2_DROPPED, index=False)

    print()
    for report in reports:
        print(report)
    print(f"\nSaved {config.STREET_GRAPH}, {config.CRIME_SNAPPED.name}, "
          f"{config.CRIME_COUNTS_MONTHLY.name}, {config.CALL_COUNTS_MONTHLY.name}")
    print(f"Saved {PHASE2_REPORT} and {PHASE2_DROPPED}")
    return G


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
    # 1. A point was dropped if its ID isn't in the snapped table.
    dropped = points.loc[~points[id_column].isin(snapped[id_column])].copy()

    # 2. How far is the nearest intersection (with no 30 m limit this time)?
    node_xy = np.array([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes])
    distance, _ = cKDTree(node_xy).query(np.column_stack([dropped.geometry.x, dropped.geometry.y]))

    # 3. One table for crimes and calls: the same columns for both.
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
