"""Tests for street_graph.py (Phase 2) on small hand-made maps.

Every test builds its own tiny map in memory, so no data files are needed and
nothing in data/processed/ is touched.

Run:  pytest tests/test_street_graph.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import Point

from saferoutes import config, street_graph

X0, Y0 = 367966.7, 5031560.6   # Elgin & Laurier, (same as test_data_pipeline.py)


def _empty_map() -> nx.MultiDiGraph:
    """A graph in metres (config.CRS_METRES), like the real one after loading."""
    return nx.MultiDiGraph(crs=config.CRS_METRES)


def _add_street(G: nx.MultiDiGraph, u: int, v: int, length: float, **tags: object) -> None:
    """Add a two-way street (one edge in each direction), like OSMnx does for walking."""
    G.add_edge(u, v, length=length, **tags)
    G.add_edge(v, u, length=length, **tags)


def _points(xy: list[tuple[float, float]], **columns: list[object]) -> gpd.GeoDataFrame:
    """A GeoDataFrame of points in metres, with any extra columns given."""
    return gpd.GeoDataFrame(columns, geometry=[Point(x, y) for x, y in xy], crs=config.CRS_METRES)


# ---------------------------------------------------------------------------
# Map: merge close intersections, keep the largest piece
# ---------------------------------------------------------------------------
@pytest.fixture
def corner_map() -> nx.MultiDiGraph:
    """One downtown corner drawn as 4 sidewalk nodes 6 m apart (1-4), four
    100 m streets leaving it (to 5-8), and a dead end (9) off node 5."""
    G = _empty_map()
    corners = {1: (0, 0), 2: (6, 0), 3: (6, 6), 4: (0, 6)}
    ends = {5: (-100, 0), 6: (106, 0), 7: (6, 106), 8: (0, -100), 9: (-100, -100)}
    for n, (x, y) in {**corners, **ends}.items():
        G.add_node(n, x=X0 + x, y=Y0 + y)
    for u, v in [(1, 2), (2, 3), (3, 4), (4, 1)]:        # crosswalks around the corner
        _add_street(G, u, v, 6.0, highway="footway", footway="crossing")
    for u, v in [(1, 5), (2, 6), (3, 7), (1, 8)]:        # streets leaving the corner
        _add_street(G, u, v, 100.0, highway="residential")
    _add_street(G, 5, 9, 100.0, highway="residential")   # dead end
    return G


def test_merge_close_intersections_makes_one_node_per_corner(
    corner_map: nx.MultiDiGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Save the merged map in a temporary folder, never in the real data/processed/.
    monkeypatch.setattr(config, "DATA_PROCESSED", tmp_path)

    merged, n_removed = street_graph.merge_close_intersections(corner_map, 10.0, use_saved=False)

    assert n_removed == 3                                     # 4 corner nodes -> 1
    assert merged.number_of_nodes() == 6                      # corner + 5 far nodes
    dead_ends = [n for n, count in merged.nodes(data="street_count") if count == 1]
    assert len(dead_ends) == 4                                # dead ends are kept, not deleted
    assert street_graph.merged_graph_path(10.0).exists()      # saved for next time


def test_merge_close_intersections_reuses_saved_file(
    corner_map: nx.MultiDiGraph, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "DATA_PROCESSED", tmp_path)
    first, _ = street_graph.merge_close_intersections(corner_map, 10.0, use_saved=False)
    second, n_removed = street_graph.merge_close_intersections(corner_map, 10.0)   # loads the file

    assert second.number_of_nodes() == first.number_of_nodes()
    assert n_removed == 3


def test_keep_largest_piece_removes_island() -> None:
    G = _empty_map()
    for n in range(1, 6):
        G.add_node(n, x=X0 + n, y=Y0)
    _add_street(G, 1, 2, 50.0)
    _add_street(G, 2, 3, 50.0)
    _add_street(G, 4, 5, 50.0)                                # island: not joined to 1-3

    kept, n_removed = street_graph.keep_largest_piece(G)

    assert set(kept.nodes) == {1, 2, 3}
    assert n_removed == 2


# ---------------------------------------------------------------------------
# Snapping and monthly counts
# ---------------------------------------------------------------------------
def test_snap_to_nodes_keeps_close_points_and_drops_far_ones() -> None:
    G = _empty_map()
    G.add_node(10, x=X0, y=Y0)
    G.add_node(20, x=X0 + 200, y=Y0)
    points = _points(
        [(X0 + 5, Y0), (X0 + 190, Y0 + 10), (X0 + 100, Y0 + 100)],
        offence_id=[1, 2, 3],
    )

    snapped, n_dropped = street_graph.snap_to_nodes(points, G, max_distance_m=30.0)

    assert list(snapped["offence_id"]) == [1, 2]
    assert list(snapped["node_id"]) == [10, 20]
    assert snapped["snap_distance_m"].iloc[0] == pytest.approx(5.0)
    assert n_dropped == 1                                     # point 3 is ~120 m from both


def test_snap_to_nodes_rejects_wrong_coordinate_system() -> None:
    G = _empty_map()
    G.add_node(10, x=X0, y=Y0)
    latlon = gpd.GeoDataFrame(geometry=[Point(-75.69, 45.42)], crs=config.CRS_LATLON)

    with pytest.raises(ValueError):
        street_graph.snap_to_nodes(latlon, G)


def test_monthly_crime_counts() -> None:
    crimes = pd.DataFrame({
        "node_id": [1, 1, 1, 2],
        "occurred_date": pd.to_datetime(["2025-03-02", "2025-03-28", "2025-04-10", "2025-03-15"]),
        "is_person": [True, False, True, False],
        "is_night": [True, False, False, False],
        "offence_group": ["person", "theft", "person", "fraud"],
    })

    counts = street_graph.monthly_crime_counts(crimes).set_index(["node_id", "month"])

    march = counts.loc[(1, pd.Timestamp("2025-03-01"))]
    assert march["n_all"] == 2
    assert march["n_person_night"] == 1
    assert march["n_theft"] == 1 and march["n_property"] == 1
    assert counts.loc[(1, pd.Timestamp("2025-04-01")), "n_person_night"] == 0    # daytime
    assert counts.loc[(2, pd.Timestamp("2025-03-01")), "n_property"] == 0        # fraud isn't property
    assert len(counts) == 3                                   # only (node, month) pairs with crime


def test_monthly_call_counts_leave_out_officer_calls_at_night() -> None:
    calls = pd.DataFrame({
        "node_id": [1, 1, 1],
        "received_date": pd.to_datetime(["2025-05-01", "2025-05-02", "2025-05-03"]),
        "initiated_by": ["citizen", "officer", "citizen"],
        "is_night": [True, True, False],
        "is_urgent": [True, True, True],
    })

    counts = street_graph.monthly_call_counts(calls).iloc[0]

    assert counts["n_calls_all"] == 3
    assert counts["n_calls_night_citizen"] == 1               # the officer call isn't counted
    assert counts["n_calls_night_urgent"] == 1


# ---------------------------------------------------------------------------
# Safety numbers on streets
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("tags", "street_type", "isolated", "underpass"),
    [
        ({"highway": "footway", "footway": "sidewalk"}, "footway", 0, 0),
        ({"highway": "footway"}, "footway", 1, 0),                         # untagged footway
        ({"highway": "service", "service": "parking_aisle"}, "service", 1, 0),
        ({"highway": "path", "tunnel": "culvert"}, "path", 1, 0),          # culvert isn't an underpass
        ({"highway": "service", "tunnel": "building_passage"}, "service", 1, 1),
        ({"highway": "residential", "covered": "yes"}, "residential", 1, 1),
        ({"highway": ["residential", "path"]}, "residential", 1, 0),       # list: any part isolated
        ({"highway": "primary"}, "primary", 0, 0),
    ],
)
def test_add_street_type_features(tags: dict, street_type: str, isolated: int, underpass: int) -> None:
    G = _empty_map()
    G.add_edge(1, 2, length=50.0, **tags)

    street_graph.add_street_type_features(G)

    data = G.edges[1, 2, 0]
    assert (data["street_type"], data["isolated"], data["underpass"]) == (street_type, isolated, underpass)


def test_add_light_features_design_doc_example() -> None:
    """Design doc, Decision 4: a 140 m street with lights at 10, 40, 70 and
    130 m has gaps 10, 30, 30, 60, 10, so the longest dark gap is 60 m."""
    G = _empty_map()
    G.add_node(1, x=X0, y=Y0)
    G.add_node(2, x=X0 + 140, y=Y0)
    G.add_node(3, x=X0 + 140, y=Y0 + 5)
    G.add_node(4, x=X0 + 600, y=Y0 + 600)
    G.add_node(5, x=X0 + 600, y=Y0 + 800)
    G.add_edge(1, 2, length=140.0)
    G.add_edge(2, 3, length=5.0)                              # short stub
    G.add_edge(4, 5, length=200.0)                            # far from every light
    lights = _points([
        (X0 + 10, Y0 + 8), (X0 + 40, Y0 - 8), (X0 + 70, Y0 + 3), (X0 + 130, Y0),
        (X0 + 130.03, Y0 + 0.02),                             # same pole recorded twice
    ])

    street_graph.add_light_features(G, lights)

    main = G.edges[1, 2, 0]
    assert main["longest_dark_gap_m"] == pytest.approx(60.0)
    assert main["lights_per_100m"] == pytest.approx(4 / 140 * 100, abs=0.001)   # duplicate counted once
    assert 0 < main["darkness"] < 0.1
    stub = G.edges[2, 3, 0]
    assert stub["lights_per_100m"] == pytest.approx(10.0)     # treated as 10 m long, not 5 m
    dark = G.edges[4, 5, 0]
    assert dark["lights_per_100m"] == 0
    assert dark["darkness"] == 1.0
    assert dark["longest_dark_gap_m"] == pytest.approx(200.0)


def test_add_crime_count_features_uses_only_crimes_before_cutoff() -> None:
    G = _empty_map()
    for n in (1, 2, 3, 4):
        G.add_node(n)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 4)
    counts = pd.DataFrame({
        "node_id": [2, 3, 3, 4, 4],
        "month": pd.to_datetime(["2025-03-01", "2025-06-01", "2025-09-01", "2025-10-01", "2021-12-01"]),
        "n_person_night": [1, 2, 4, 50, 70],   # node 4: on the cut-off month, and too old
    })

    street_graph.add_crime_count_features(G, counts, date(2025, 10, 1))

    past = [G.edges[u, v, 0]["past_crimes"] for u, v in [(1, 2), (2, 3), (3, 4)]]
    assert past == [0.5, 3.5, 3.0]                            # node 4's crimes are not used
    average = (0.5 + 3.5 + 3.0) / 3                           # option C: streets that had crime
    assert G.edges[2, 3, 0]["crime_score"] == pytest.approx(3.5 / average, abs=0.001)
    assert G.graph["crime_cutoff"] == "2025-10-01"


def test_add_crime_count_features_caps_the_score() -> None:
    G = _empty_map()
    for n in range(1, 13):
        G.add_node(n)
    for n in range(1, 12):
        G.add_edge(n, n + 1)
    counts = pd.DataFrame({                                   # one hot corner, ten quiet ones
        "node_id": [1] + list(range(3, 13)),
        "month": pd.Timestamp("2025-05-01"),
        "n_person_night": [1000] + [1] * 10,
    })

    street_graph.add_crime_count_features(G, counts, date(2025, 10, 1))

    assert G.edges[1, 2, 0]["crime_score"] == config.CRIME_SCORE_CAP


# ---------------------------------------------------------------------------
# Save and load
# ---------------------------------------------------------------------------
def test_save_and_load_keep_numbers_as_numbers(tmp_path: Path) -> None:
    G = _empty_map()
    G.add_node(1, x=X0, y=Y0, street_count=1)
    G.add_node(2, x=X0 + 100, y=Y0, street_count=1)
    G.add_edge(1, 2, length=100.0, highway=["residential", "path"], street_type="residential",
               isolated=1, underpass=0, lights_per_100m=2.5, longest_dark_gap_m=40.0,
               darkness=0.167, past_crimes=0.5, crime_score=0.54)
    G.graph["crime_cutoff"] = "2025-10-01"
    G.graph["crime_score_average"] = 0.92
    path = tmp_path / "street_graph.graphml"

    street_graph.save_graph(G, path)
    loaded = street_graph.load_graph(path)

    data = loaded.edges[1, 2, 0]
    assert data["darkness"] == pytest.approx(0.167) and isinstance(data["darkness"], float)
    assert data["isolated"] == 1 and isinstance(data["isolated"], int)
    assert data["crime_score"] == pytest.approx(0.54)
    assert data["highway"] == ["residential", "path"]         # lists come back as lists
    assert loaded.graph["crime_score_average"] == pytest.approx(0.92)


def test_load_graph_missing_file_says_what_to_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="02_build_graph.py"):
        street_graph.load_graph(tmp_path / "missing.graphml")
