"""Shared settings for the whole project.

Every decision from the project plan lives here, so nobody hard-codes a
different value in their own file. Change a value here, not in your code.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parents[2]
DATA_RAW: Path = ROOT / "data" / "raw"
DATA_PROCESSED: Path = ROOT / "data" / "processed"
RESULTS: Path = ROOT / "results"

# ---------------------------------------------------------------------------
# Raw files (downloaded by hand, see data/README.md)
# ---------------------------------------------------------------------------
RAW_CRIME: Path = DATA_RAW / "Criminal_Offences_Open_Data.csv"
RAW_LIGHTS: Path = DATA_RAW / "Street_Lights.csv"
RAW_OSM_GRAPH: Path = DATA_RAW / "osm_walk_raw.graphml"    # made by scripts/00_download_osm.py
RAW_OSM_AREAS: Path = DATA_RAW / "osm_areas.parquet"       # optional, same script
RAW_TRANSIT_STOPS: Path = DATA_RAW / "OC_Transpo_Stops.csv"
RAW_GTFS_DIR: Path = DATA_RAW / "gtfs"                    # optional, future: buses per hour at night
RAW_VENUES: Path = DATA_RAW / "agco_liquor_licences.csv"  # optional

# ---------------------------------------------------------------------------
# Shared hand-off files (the "contracts" between team members)
# Column / attribute names are listed next to each file.
# ---------------------------------------------------------------------------
# Phase 1 -> everyone
CRIME_CLEAN: Path = DATA_PROCESSED / "crime_clean.parquet"
#   offence_id (int), occurred_date (datetime), reported_date (datetime),
#   hour (int 0-23), offence_summary (str), offence_category (str),
#   offence_group (str, see OFFENCE_GROUPS), is_person (bool), is_night (bool),
#   hour_unknown (bool), neighbourhood (str), intersection (str),
#   geometry (Point, CRS_METRES)
LIGHTS_CLEAN: Path = DATA_PROCESSED / "lights_clean.parquet"
#   light_id (int), light_type (str), n_heads (int), wattage (int),
#   geometry (Point, CRS_METRES)
STOPS_CLEAN: Path = DATA_PROCESSED / "stops_clean.parquet"
#   stop_id (int), stop_code (str or missing), stop_name (str),
#   has_shelter (bool), has_bench (bool), geometry (Point, CRS_METRES)
CLEANING_REPORT: Path = RESULTS / "cleaning_report.csv"
#   one row per (dataset, item): rows in/out, rows dropped per reason, notes

# Phase 2 -> ML, routing, evaluation
CRIME_SNAPPED: Path = DATA_PROCESSED / "crime_snapped.parquet"
#   all CRIME_CLEAN columns + node_id (int), snap_distance_m (float)
CRIME_COUNTS_MONTHLY: Path = DATA_PROCESSED / "crime_counts_monthly.parquet"
#   node_id (int), month (datetime, first day of month),
#   n_person_night, n_all, n_property, n_mischief, n_theft (int)
STREET_GRAPH: Path = DATA_PROCESSED / "street_graph.graphml"
#   edge attributes: see EDGE_ATTRIBUTES below

# Phase 3 -> routing, evaluation, demo
TRAINING_TABLE: Path = DATA_PROCESSED / "training_table.parquet"
#   node_id, cutoff, <feature columns>, label (0/1), n_future (int)
NODE_RISK_EVAL: Path = DATA_PROCESSED / "node_risk_eval.parquet"   # cut-off = TEST_CUTOFF
NODE_RISK_DEMO: Path = DATA_PROCESSED / "node_risk_demo.parquet"   # cut-off = DEMO_CUTOFF
#   node_id (int), probability (float 0-1), risk_score (float, 1 = city average)
MODEL_FILE: Path = DATA_PROCESSED / "risk_model.joblib"

# Edge attributes every street in STREET_GRAPH must carry
EDGE_ATTRIBUTES: tuple[str, ...] = (
    "length",            # metres (from OSMnx)
    "lights_per_100m",   # float
    "longest_dark_gap_m",
    "darkness",          # float 0-1, 1 = no lights
    "street_type",       # OSM highway tag, e.g. "primary", "footway"
    "isolated",          # 0/1: park path, alley, parking lot, industrial
    "underpass",         # 0/1: tunnel or covered
    "past_crimes_12m",   # avg of the two end intersections (fallback crime number)
    "crime_score",       # past_crimes_12m / city average (Route 2)
    "model_risk_score",  # ML risk, avg of two ends / city average (Route 3)
)

# ---------------------------------------------------------------------------
# Coordinate systems
# ---------------------------------------------------------------------------
CRS_METRES: str = "EPSG:2951"   # MTM zone 9 (metres). Everything is converted to this.
CRS_CRIME_RAW: str = "EPSG:2951"
CRS_LIGHTS_RAW: str = "EPSG:3857"  # Web Mercator
CRS_STOPS_RAW: str = "EPSG:4326"   # bus stops: Latitude / Longitude columns
CRS_LATLON: str = "EPSG:4326"

# ---------------------------------------------------------------------------
# Study area
# ---------------------------------------------------------------------------
STUDY_AREA: str = "Ottawa, Ontario, Canada"   # narrowed to urban Ottawa in Phase 2
DEMO_AREA: tuple[str, ...] = ("Centretown", "Byward Market", "Lowertown")

# OpenStreetMap download area: (west, south, east, north) in longitude/latitude.
# Default = urban Ottawa inside the Greenbelt (downtown + inner suburbs;
# Kanata, Barrhaven and Orleans are NOT included). Change here if the team
# picks a different study area, then re-run scripts/00_download_osm.py.
OSM_BBOX_LATLON: tuple[float, float, float, float] = (-75.85, 45.33, -75.58, 45.47)

# Extra street tags to keep (OSMnx keeps highway, tunnel, bridge, service, ... by default)
OSM_EXTRA_WAY_TAGS: tuple[str, ...] = ("covered", "footway", "lit")

# Area shapes for the optional isolation check (water, parks, parking, industrial)
OSM_AREA_TAGS: dict[str, bool | str | list[str]] = {
    "waterway": True,
    "natural": "water",
    "leisure": "park",
    "amenity": "parking",
    "landuse": "industrial",
}

# Box around the City of Ottawa in CRS_METRES: (min_x, min_y, max_x, max_y).
# Points outside it are location errors and are dropped.
OTTAWA_BOUNDS_M: tuple[float, float, float, float] = (315_000.0, 4_979_000.0, 404_000.0, 5_046_000.0)

# ---------------------------------------------------------------------------
# Crime definitions
# ---------------------------------------------------------------------------
# The police file covers crimes REPORTED in 2018-2025. Some occurred much
# earlier (back to 1948); those are dropped because years before 2018 are
# incomplete.
DATA_START: date = date(2018, 1, 1)
DATA_END: date = date(2025, 12, 31)

PERSON_CRIME_PREFIX: str = "Crimes Against The Person"
NIGHT_HOURS: frozenset[int] = frozenset({18, 19, 20, 21, 22, 23, 1, 2, 3, 4, 5})
UNKNOWN_HOUR: int = 0   # hour 0 looks like "time unknown"; never counted as night

# Offence Category -> our group. Used for the activity features in Phase 3.
# Anything not listed becomes "other".
OFFENCE_GROUPS: dict[str, str] = {
    # crimes against the person
    "Assaults": "person",
    "Attempting The Commission Of A Capital Crime": "person",
    "Commodification Of Sexual Activity": "person",
    "Other Violations Involving Violence Or The Threat Of Violence": "person",
    "Sexual Violations": "person",
    "Violations Causing Death": "person",
    "Violations Resulting In The Deprivation Of Freedom": "person",
    # property
    "Theft $5000 and Under": "theft",
    "Theft Over $5000": "theft",
    "Theft - Motor Vehicle": "theft",
    "Possession / Trafficking Stolen Goods": "theft",
    "Mischief": "mischief",
    "Arson": "mischief",
    "Break and Enter": "break_and_enter",
    "Fraud": "fraud",          # often recorded at a home or police station, not on the street
}

# ---------------------------------------------------------------------------
# Distances (metres)
# ---------------------------------------------------------------------------
SNAP_MAX_DISTANCE_M: float = 30.0      # crime -> nearest intersection
LIGHT_BUFFER_M: float = 25.0           # lights counted for a street
LIGHTS_NEAR_NODE_M: float = 50.0       # lights counted for an intersection
STOPS_NEAR_NODE_M: float = 200.0       # bus stops counted for an intersection
NEIGHBOUR_RADIUS_M: float = 250.0      # neighbouring intersections
DARK_THRESHOLD_LIGHTS_PER_100M: float = 3.0   # >= this means darkness = 0

# About 6,000 light records share a position with another record: two lights
# on one pole, exact copies, or an old light that was replaced but never
# removed. lights_clean keeps every record; Phase 2 counts each position ONCE
# (positions rounded to this precision) for lights_per_100m and dark gaps.
LIGHT_POSITION_ROUND_M: float = 0.1
CRIME_NEAR_ROUTE_M: float = 50.0       # evaluation: crimes "passed" by a route

# ---------------------------------------------------------------------------
# Time split (Phase 3)
# ---------------------------------------------------------------------------
LABEL_MONTHS: int = 3
TRAIN_FIRST_CUTOFF: date = date(2022, 1, 1)
TRAIN_LAST_CUTOFF: date = date(2025, 4, 1)   # labels end Jun 30, 2025
VALIDATION_CUTOFF: date = date(2025, 7, 1)   # labels Jul-Sep 2025
TEST_CUTOFF: date = date(2025, 10, 1)        # labels Oct-Dec 2025, used ONCE
DEMO_CUTOFF: date = date(2026, 1, 1)         # "current" risk for the demo

# ---------------------------------------------------------------------------
# Routing (Phase 5)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS: dict[str, float] = {"crime": 0.5, "dark": 0.5, "isolated": 0.5}
MAX_DETOUR_RATIO: float = 1.25   # target when tuning weights

# ---------------------------------------------------------------------------
# Evaluation (Phase 6)
# ---------------------------------------------------------------------------
EVAL_N_TRIPS: int = 100
TRIP_MIN_M: float = 500.0
TRIP_MAX_M: float = 3000.0
RANDOM_SEED: int = 42
