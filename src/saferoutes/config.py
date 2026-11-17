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
RAW_GTFS_DIR: Path = DATA_RAW / "gtfs"                    # optional
RAW_VENUES: Path = DATA_RAW / "agco_liquor_licences.csv"  # optional

# ---------------------------------------------------------------------------
# Shared hand-off files (the "contracts" between team members)
# Column / attribute names are listed next to each file.
# ---------------------------------------------------------------------------
# Phase 1 -> everyone
CRIME_CLEAN: Path = DATA_PROCESSED / "crime_clean.parquet"
#   occurred_date (datetime), hour (int 0-23), offence_summary (str),
#   offence_category (str), is_person (bool), is_night (bool),
#   hour_unknown (bool), geometry (Point, CRS_METRES)
LIGHTS_CLEAN: Path = DATA_PROCESSED / "lights_clean.parquet"
#   light_id (int), light_type (str), geometry (Point, CRS_METRES)

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
CRS_LATLON: str = "EPSG:4326"

# ---------------------------------------------------------------------------
# Study area
# ---------------------------------------------------------------------------
STUDY_AREA: str = "Ottawa, Ontario, Canada"   # narrowed to urban Ottawa in Phase 2
DEMO_AREA: tuple[str, ...] = ("Centretown", "Byward Market", "Lowertown")

# ---------------------------------------------------------------------------
# Crime definitions
# ---------------------------------------------------------------------------
PERSON_CRIME_PREFIX: str = "Crimes Against The Person"
NIGHT_HOURS: frozenset[int] = frozenset({18, 19, 20, 21, 22, 23, 1, 2, 3, 4, 5})
UNKNOWN_HOUR: int = 0   # hour 0 looks like "time unknown"; never counted as night

# ---------------------------------------------------------------------------
# Distances (metres)
# ---------------------------------------------------------------------------
SNAP_MAX_DISTANCE_M: float = 30.0      # crime -> nearest intersection
LIGHT_BUFFER_M: float = 25.0           # lights counted for a street
LIGHTS_NEAR_NODE_M: float = 50.0       # lights counted for an intersection
NEIGHBOUR_RADIUS_M: float = 250.0      # neighbouring intersections
DARK_THRESHOLD_LIGHTS_PER_100M: float = 3.0   # >= this means darkness = 0
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
