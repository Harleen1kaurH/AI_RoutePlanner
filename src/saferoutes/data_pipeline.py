"""Phase 1: load and clean the raw data.

Owner: Data person

Output files (see config.py for the exact columns):
    config.CRIME_CLEAN, config.LIGHTS_CLEAN
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd

from saferoutes import config


@dataclass
class CleaningReport:
    """How many rows came in, went out, and why rows were dropped.

    Goes into the report's Dataset section.
    """

    name: str
    rows_in: int
    rows_out: int
    dropped: dict[str, int] = field(default_factory=dict)  # reason -> count


# ---------------------------------------------------------------------------
# Crime
# ---------------------------------------------------------------------------
def load_raw_crime(path: Path = config.RAW_CRIME) -> pd.DataFrame:
    """Read the Ottawa Police criminal offences CSV exactly as downloaded."""
    raise NotImplementedError


def clean_crime(raw: pd.DataFrame) -> tuple[gpd.GeoDataFrame, CleaningReport]:
    """Clean the raw crime table.

    Steps:
        - use "Occurred Date" (not "Reported Date") -> occurred_date
        - convert "Occurred Hour" (e.g. 2300) -> hour (0-23)
        - is_person = Offence Summary starts with config.PERSON_CRIME_PREFIX
        - hour_unknown = hour == config.UNKNOWN_HOUR
        - is_night = hour in config.NIGHT_HOURS (never true when hour_unknown)
        - drop rows with no x/y; build Point geometry in config.CRS_METRES

    Returns the clean table (columns listed at config.CRIME_CLEAN) and a report.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Street lights
# ---------------------------------------------------------------------------
def load_raw_lights(path: Path = config.RAW_LIGHTS) -> pd.DataFrame:
    """Read the City of Ottawa street lights CSV exactly as downloaded."""
    raise NotImplementedError


def clean_lights(raw: pd.DataFrame) -> tuple[gpd.GeoDataFrame, CleaningReport]:
    """Drop lights with no location and convert from config.CRS_LIGHTS_RAW
    to config.CRS_METRES.

    Returns columns: light_id, light_type, geometry.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Optional sources (only if time allows)
# ---------------------------------------------------------------------------
def load_transit_stops(gtfs_dir: Path = config.RAW_GTFS_DIR) -> gpd.GeoDataFrame:
    """OPTIONAL. OC Transpo stops with the number of trips after 10 pm.

    Returns columns: stop_id, night_trips_per_hour, geometry.
    """
    raise NotImplementedError


def load_licensed_venues(path: Path = config.RAW_VENUES) -> gpd.GeoDataFrame:
    """OPTIONAL. AGCO liquor licences, geocoded to points.

    Returns columns: venue_id, licence_type, geometry.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_parquet(table: pd.DataFrame | gpd.GeoDataFrame, path: Path) -> None:
    """Save a table as Parquet, creating the folder if needed."""
    raise NotImplementedError


def run() -> list[CleaningReport]:
    """Run all of Phase 1: load, clean and save crime and lights.

    Returns the cleaning reports so the script can print them.
    """
    raise NotImplementedError
