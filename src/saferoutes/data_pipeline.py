"""Phase 1: load and clean the raw data.

Owner: Data person

Output files (see config.py for the exact columns):
    config.CRIME_CLEAN, config.LIGHTS_CLEAN, config.STOPS_CLEAN,
    config.CALLS_CLEAN, config.VENUES_CLEAN, config.CLEANING_REPORT

Phase 1 has two steps:
    1. scripts/00_download_osm.py (needs internet, run once): makes the study
       area (make_study_area below) and downloads the OpenStreetMap map, bars
       and areas inside it.
    2. scripts/01_prepare_data.py (this file's run()): cleans every dataset and
       drops everything outside the study area, so every table covers exactly
       the same area as the map.

The map itself is loaded in Phase 2 (street_graph.load_osm_walk_graph).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from saferoutes import config

# Raw crime column -> our column name. Only these columns are kept.
RAW_CRIME_COLUMNS: dict[str, str] = {
    "OBJECTID": "offence_id",
    "Occurred Date": "occurred_date",
    "Occurred Hour": "occurred_hour",
    "Reported Date": "reported_date",
    "Offence Summary": "offence_summary",
    "Offence Category": "offence_category",
    "Neighbourhood Name": "neighbourhood",
    "Intersection": "intersection",
    "x": "x",
    "y": "y",
}

# Raw street-light column -> our column name.
RAW_LIGHT_COLUMNS: dict[str, str] = {
    "OBJECTID": "light_id",
    "LIGHT_TYPE": "light_type",
    "LIGHTS_NUM": "n_heads",
    "WATTAGE": "wattage",
    "X": "x",
    "Y": "y",
}

# Raw bus-stop column -> our column name.
RAW_STOP_COLUMNS: dict[str, str] = {
    "FID": "stop_id",
    "F560": "stop_code",      # the number shown at the stop; "0" = no code
    "Location": "stop_name",  # e.g. "SUSSEX / RIDEAU FALLS"
    "Shelter": "shelter",     # "Y" or blank
    "Bench": "bench",         # "Y" or blank
    "Latitude": "lat",
    "Longitude": "lon",
}

# Raw calls-for-service column -> our column name.
RAW_CALL_COLUMNS: dict[str, str] = {
    "OBJECTID": "call_id",
    "Received Date": "received_date",
    "Received Hour": "received_hour",
    "Priority": "priority",           # 1 (most urgent) to 7
    "Initiated By": "initiated_by",   # C = citizen called, O = officer started it
    "Neighbourhood": "neighbourhood",
    "Intersection": "intersection",
    "x": "x",
    "y": "y",
}

# A crime reported this many days after it happened counts as "reported late"
# (a note for the report, not a reason to drop it).
LATE_REPORT_DAYS: int = 90


@dataclass
class CleaningReport:
    """How many rows came in, went out, and why rows were dropped.

    Goes into the report's Dataset section.
    """

    name: str
    rows_in: int
    rows_out: int
    dropped: dict[str, int] = field(default_factory=dict)  # reason -> rows dropped
    notes: dict[str, int] = field(default_factory=dict)    # useful counts, nothing dropped

    def __str__(self) -> str:
        """Readable summary for printing: rows in -> out, then one line per
        drop reason and per note."""
        lines = [f"== {self.name}: {self.rows_in:,} rows in -> {self.rows_out:,} rows out"]
        for reason, n in self.dropped.items():
            lines.append(f"   dropped {n:>8,}  {reason}")
        for label, n in self.notes.items():
            lines.append(f"   note    {n:>8,}  {label}")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """One row per item, for saving to CSV."""
        rows = [{"dataset": self.name, "kind": "rows_in", "item": "", "count": self.rows_in}]
        rows += [{"dataset": self.name, "kind": "dropped", "item": k, "count": v} for k, v in self.dropped.items()]
        rows += [{"dataset": self.name, "kind": "note", "item": k, "count": v} for k, v in self.notes.items()]
        rows.append({"dataset": self.name, "kind": "rows_out", "item": "", "count": self.rows_out})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _find_raw_file(path: Path, pattern: str) -> Path:
    """Return `path` if it exists, otherwise the single file in the same folder
    matching `pattern` (downloads often add a number to the file name)."""
    if path.exists():
        return path
    matches = sorted(path.parent.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find {path.name} in {path.parent}. "
        f"Download it and save it there (see data/README.md). "
        f"Files matching '{pattern}': {[m.name for m in matches]}"
    )


def _check_columns(table: pd.DataFrame, needed: list[str], name: str) -> None:
    """Stop with a clear error if any column in `needed` is missing from `table`.

    Catches a changed download format right away, instead of a confusing
    error later. `name` is used in the message, e.g. "Crime file".
    """
    missing = [c for c in needed if c not in table.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}. Has the download format changed?")


def _drop(table: pd.DataFrame, remove: pd.Series, reason: str, dropped: dict[str, int]) -> pd.DataFrame:
    """Drop rows where `remove` is True and record how many under `reason`."""
    n = int(remove.sum())
    if n:
        dropped[reason] = n
    return table.loc[~remove]


def _outside_ottawa(x: pd.Series, y: pd.Series) -> pd.Series:
    """True for every point outside config.OTTAWA_BOUNDS_M.

    x and y must already be in config.CRS_METRES. Used to drop location errors
    (e.g. a light recorded near the equator).
    """
    min_x, min_y, max_x, max_y = config.OTTAWA_BOUNDS_M
    return ~(x.between(min_x, max_x) & y.between(min_y, max_y))


def _outside_area(points: gpd.GeoSeries, area: BaseGeometry) -> pd.Series:
    """True for every point outside `area` (both in config.CRS_METRES).

    Uses shapely's fast point-in-polygon test, so 1 million calls take seconds.
    """
    shapely.prepare(area)
    inside = shapely.contains_xy(area, points.x.to_numpy(), points.y.to_numpy())
    return pd.Series(~inside, index=points.index)


def _drop_outside_study_area(
    table: gpd.GeoDataFrame, study_area: BaseGeometry | None, dropped: dict[str, int]
) -> gpd.GeoDataFrame:
    """Drop rows outside the study area (if one is given) and record how many.

    study_area=None keeps everything; the unit tests use that.
    """
    if study_area is None:
        return table
    return _drop(table, _outside_area(table.geometry, study_area), "outside the study area", dropped)


def _parse_day(text: pd.Series) -> pd.Series:
    """'1/24/2018 5:00:00 AM' -> 2018-01-24.

    The time part is always 5:00 AM (a time-zone artifact), so only the
    date is kept. The real hour comes from the "Occurred Hour" column.
    """
    return pd.to_datetime(text.str.split(" ").str[0], format="%m/%d/%Y", errors="coerce")


# ---------------------------------------------------------------------------
# Study area
# ---------------------------------------------------------------------------
def make_study_area(
    city_boundary: gpd.GeoDataFrame,
    bbox: tuple[float, float, float, float] = config.STUDY_BBOX_LATLON,
    margin_m: float = config.STUDY_MARGIN_M,
) -> gpd.GeoDataFrame:
    """Build the study area: the team's box, widened by margin_m, cut to the City.

    bbox is (west, south, east, north) in longitude/latitude. city_boundary is
    the official City of Ottawa shape (00_download_osm.py gets it from
    OpenStreetMap with ox.geocode_to_gdf), in any coordinate system.

    Steps:
        1. box -> metres (config.CRS_METRES)
        2. widen by margin_m on every side (square corners), so roads running
           close to the edge stay connected instead of becoming islands
        3. keep only the part inside the City; this removes Gatineau exactly
           along the provincial border in the middle of the river

    Returns one row: name, area_km2, geometry (in config.CRS_METRES).
    """
    box = gpd.GeoSeries([shapely.box(*bbox)], crs=config.CRS_LATLON).to_crs(config.CRS_METRES)
    wide_box = box.iloc[0].buffer(margin_m, join_style="mitre")
    city = city_boundary.to_crs(config.CRS_METRES).union_all()
    area = shapely.make_valid(wide_box.intersection(city))
    if area.is_empty:
        raise ValueError("The study box and the City boundary don't overlap. Check STUDY_BBOX_LATLON.")
    return gpd.GeoDataFrame(
        {"name": ["study area"], "area_km2": [area.area / 1e6]},
        geometry=[area],
        crs=config.CRS_METRES,
    )


def load_study_area(path: Path = config.STUDY_AREA) -> BaseGeometry:
    """Read the study area saved by scripts/00_download_osm.py, as one shape in
    config.CRS_METRES."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found in {path.parent}. "
            "Run first: python scripts/00_download_osm.py --venues --areas   (needs internet)"
        )
    return gpd.read_parquet(path).to_crs(config.CRS_METRES).union_all()


# ---------------------------------------------------------------------------
# Crime
# ---------------------------------------------------------------------------
def load_raw_crime(path: Path = config.RAW_CRIME) -> pd.DataFrame:
    """Read the Ottawa Police criminal offences CSV exactly as downloaded."""
    path = _find_raw_file(path, "Criminal_Offences_Open_Data*.csv")
    raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    _check_columns(raw, list(RAW_CRIME_COLUMNS), "Crime file")
    return raw


def clean_crime(
    raw: pd.DataFrame, study_area: BaseGeometry | None = None
) -> tuple[gpd.GeoDataFrame, CleaningReport]:
    """Clean the raw crime table.

    Steps:
        - use "Occurred Date" (not "Reported Date") -> occurred_date
        - convert "Occurred Hour" (e.g. 2300) -> hour (0-23)
        - is_person = Offence Summary starts with config.PERSON_CRIME_PREFIX
        - hour_unknown = hour == config.UNKNOWN_HOUR
        - is_night = hour in config.NIGHT_HOURS (never true when hour_unknown)
        - drop rows with no x/y; build Point geometry in config.CRS_METRES
        - drop crimes outside study_area (if given)

    Returns the clean table (columns listed at config.CRIME_CLEAN) and a report.
    """
    _check_columns(raw, list(RAW_CRIME_COLUMNS), "Crime file")
    rows_in = len(raw)
    dropped: dict[str, int] = {}

    df = raw[list(RAW_CRIME_COLUMNS)].rename(columns=RAW_CRIME_COLUMNS)

    # 1. Dates: occurred date is the one we use
    df["occurred_date"] = _parse_day(df["occurred_date"].astype(str))
    df["reported_date"] = _parse_day(df["reported_date"].astype(str))
    df = _drop(df, df["occurred_date"].isna(), "occurred date missing or unreadable", dropped)

    start, end = pd.Timestamp(config.DATA_START), pd.Timestamp(config.DATA_END)
    df = _drop(df, df["occurred_date"] < start,
               f"occurred before {config.DATA_START} (outside dataset coverage)", dropped)
    df = _drop(df, df["occurred_date"] > end, f"occurred after {config.DATA_END}", dropped)

    # 2. Hour: stored as 0, 100, ..., 2300
    hour_raw = pd.to_numeric(df["occurred_hour"], errors="coerce")
    bad_hour = hour_raw.isna() | (hour_raw % 100 != 0) | ~hour_raw.between(0, 2300)
    df = _drop(df, bad_hour, "occurred hour missing or not a whole hour", dropped)
    df["hour"] = (pd.to_numeric(df["occurred_hour"]) // 100).astype(int)

    # 3. Location
    df = _drop(df, df["x"].isna() | df["y"].isna(), "no location (x/y missing)", dropped)
    df = _drop(df, _outside_ottawa(df["x"], df["y"]), "location outside Ottawa", dropped)

    # 4. Flags
    df["offence_group"] = df["offence_category"].map(config.OFFENCE_GROUPS).fillna("other")
    df["is_person"] = df["offence_summary"].str.startswith(config.PERSON_CRIME_PREFIX, na=False)
    df["hour_unknown"] = df["hour"] == config.UNKNOWN_HOUR
    df["is_night"] = df["hour"].isin(config.NIGHT_HOURS) & ~df["hour_unknown"]

    # 5. Geometry in metres (crime x/y are already MTM zone 9; to_crs is a
    #    safety net in case config.CRS_METRES is ever changed)
    clean = gpd.GeoDataFrame(
        df.drop(columns=["occurred_hour", "x", "y"]),
        geometry=gpd.points_from_xy(df["x"], df["y"]),
        crs=config.CRS_CRIME_RAW,
    ).to_crs(config.CRS_METRES)
    clean = _drop_outside_study_area(clean, study_area, dropped)

    columns = ["offence_id", "occurred_date", "reported_date", "hour", "offence_summary",
               "offence_category", "offence_group", "is_person", "is_night", "hour_unknown",
               "neighbourhood", "intersection", "geometry"]
    clean = clean[columns].reset_index(drop=True)

    delay_days = (clean["reported_date"] - clean["occurred_date"]).dt.days
    notes = {
        "crimes against the person": int(clean["is_person"].sum()),
        "night-time crimes against the person (the label)": int((clean["is_person"] & clean["is_night"]).sum()),
        "hour 0, treated as time unknown (kept, never night)": int(clean["hour_unknown"].sum()),
        f"reported more than {LATE_REPORT_DAYS} days after occurring": int((delay_days > LATE_REPORT_DAYS).sum()),
        "no intersection name (location still present)": int(clean["intersection"].isna().sum()),
    }
    report = CleaningReport("crime", rows_in, len(clean), dropped, notes)
    assert report.rows_in - sum(dropped.values()) == report.rows_out
    return clean, report


# ---------------------------------------------------------------------------
# Street lights
# ---------------------------------------------------------------------------
def load_raw_lights(path: Path = config.RAW_LIGHTS) -> pd.DataFrame:
    """Read the City of Ottawa street lights CSV exactly as downloaded."""
    path = _find_raw_file(path, "Street_Lights*.csv")
    raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    _check_columns(raw, list(RAW_LIGHT_COLUMNS), "Street lights file")
    return raw


def clean_lights(
    raw: pd.DataFrame, study_area: BaseGeometry | None = None
) -> tuple[gpd.GeoDataFrame, CleaningReport]:
    """Drop lights with no location and convert from config.CRS_LIGHTS_RAW
    to config.CRS_METRES, then drop lights outside study_area (if given).

    Returns columns: light_id, light_type, n_heads, wattage, geometry.
    Several records can share one position (one pole, several fixtures);
    they are kept and counted in the notes.
    """
    _check_columns(raw, list(RAW_LIGHT_COLUMNS), "Street lights file")
    rows_in = len(raw)
    dropped: dict[str, int] = {}

    df = raw[list(RAW_LIGHT_COLUMNS)].rename(columns=RAW_LIGHT_COLUMNS)
    df = _drop(df, df["x"].isna() | df["y"].isna(), "no location (X/Y missing)", dropped)

    lights = gpd.GeoDataFrame(
        df.drop(columns=["x", "y"]),
        geometry=gpd.points_from_xy(df["x"], df["y"]),
        crs=config.CRS_LIGHTS_RAW,
    ).to_crs(config.CRS_METRES)

    outside = _outside_ottawa(lights.geometry.x, lights.geometry.y)
    lights = _drop(lights, outside, "location outside Ottawa", dropped)
    lights = _drop_outside_study_area(lights, study_area, dropped)

    lights["n_heads"] = pd.to_numeric(lights["n_heads"], errors="coerce").fillna(1).astype(int)
    lights["wattage"] = pd.to_numeric(lights["wattage"], errors="coerce").fillna(0).astype(int)
    lights = lights[["light_id", "light_type", "n_heads", "wattage", "geometry"]].reset_index(drop=True)

    xy = pd.DataFrame({"x": lights.geometry.x.round(1), "y": lights.geometry.y.round(1)})
    notes = {"records sharing a position with another record (kept)": int(xy.duplicated().sum())}
    report = CleaningReport("street lights", rows_in, len(lights), dropped, notes)
    assert report.rows_in - sum(dropped.values()) == report.rows_out
    return lights, report


# ---------------------------------------------------------------------------
# Bus stops
# ---------------------------------------------------------------------------
def load_raw_stops(path: Path = config.RAW_TRANSIT_STOPS) -> pd.DataFrame:
    """Read the OC Transpo stops CSV exactly as downloaded.

    Stop codes (column F560) are read as text, so codes like "0123" keep their zero.
    """
    path = _find_raw_file(path, "OC_Transpo_Stops*.csv")
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype={"F560": str})
    _check_columns(raw, list(RAW_STOP_COLUMNS), "Bus stops file")
    return raw


def clean_stops(
    raw: pd.DataFrame, study_area: BaseGeometry | None = None
) -> tuple[gpd.GeoDataFrame, CleaningReport]:
    """Clean the OC Transpo stops table.

    Steps:
        - stop code "0" means "no code", so it becomes missing
        - Shelter / Bench are "Y" or blank -> has_shelter / has_bench (True/False)
        - drop rows with no latitude/longitude
        - convert from latitude/longitude (config.CRS_STOPS_RAW) to config.CRS_METRES
        - drop stops outside the Ottawa box, then outside study_area (if given)

    The file has stop locations only, no timetable, so it can't say how many
    buses stop at night. That needs the OC Transpo GTFS timetable (future work).

    Returns columns: stop_id, stop_code, stop_name, has_shelter, has_bench, geometry.
    """
    _check_columns(raw, list(RAW_STOP_COLUMNS), "Bus stops file")
    rows_in = len(raw)
    dropped: dict[str, int] = {}

    df = raw[list(RAW_STOP_COLUMNS)].rename(columns=RAW_STOP_COLUMNS)
    df = _drop(df, df["lat"].isna() | df["lon"].isna(), "no location (Latitude/Longitude missing)", dropped)

    df["stop_code"] = df["stop_code"].astype("string").str.strip()
    df.loc[df["stop_code"].isin(["", "0"]), "stop_code"] = pd.NA
    df["stop_name"] = df["stop_name"].astype("string").str.strip()
    df["has_shelter"] = df["shelter"].astype("string").str.strip().str.upper().eq("Y").fillna(False).astype(bool)
    df["has_bench"] = df["bench"].astype("string").str.strip().str.upper().eq("Y").fillna(False).astype(bool)

    stops = gpd.GeoDataFrame(
        df.drop(columns=["lat", "lon", "shelter", "bench"]),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),   # x = longitude, y = latitude
        crs=config.CRS_STOPS_RAW,
    ).to_crs(config.CRS_METRES)

    outside = _outside_ottawa(stops.geometry.x, stops.geometry.y)
    stops = _drop(stops, outside, "location outside Ottawa", dropped)
    stops = _drop_outside_study_area(stops, study_area, dropped)

    stops = stops[["stop_id", "stop_code", "stop_name", "has_shelter", "has_bench", "geometry"]]
    stops = stops.reset_index(drop=True)

    codes = stops["stop_code"].dropna()
    notes = {
        "stops with a shelter": int(stops["has_shelter"].sum()),
        "stops with a bench": int(stops["has_bench"].sum()),
        "stops with no stop code (kept)": int(stops["stop_code"].isna().sum()),
        "stops sharing a stop code with another stop (kept)": int(codes.duplicated(keep=False).sum()),
    }
    report = CleaningReport("bus stops", rows_in, len(stops), dropped, notes)
    assert report.rows_in - sum(dropped.values()) == report.rows_out
    return stops, report


# ---------------------------------------------------------------------------
# Calls for service
# ---------------------------------------------------------------------------
def load_raw_calls(path: Path = config.RAW_CALLS) -> pd.DataFrame:
    """Read the Ottawa Police dispatched calls for service CSV as downloaded
    (about 1.1 million rows, 2021-2025)."""
    path = _find_raw_file(path, "Calls_For_Service*.csv")
    raw = pd.read_csv(path, encoding="utf-8-sig", usecols=list(RAW_CALL_COLUMNS), low_memory=False)
    _check_columns(raw, list(RAW_CALL_COLUMNS), "Calls for service file")
    return raw


def clean_calls(
    raw: pd.DataFrame, study_area: BaseGeometry | None = None
) -> tuple[gpd.GeoDataFrame, CleaningReport]:
    """Clean the calls for service table.

    Calls have NO call type (we can't tell a fight from a noise complaint),
    so they're used as features about how often police are needed somewhere,
    never as the label.

    Steps:
        - "Received Date" -> received_date (date part only; the time is always 5:00 AM)
        - "Received Hour" (e.g. 2200) -> hour (0-23). Hour 0 is real midnight
          here, so is_night uses config.CALLS_NIGHT_HOURS (includes 0)
        - Initiated By "C" -> "citizen", "O" -> "officer"
        - is_urgent = priority <= config.URGENT_PRIORITY_MAX
        - drop rows with no x/y; convert from Web Mercator to config.CRS_METRES
        - drop calls outside study_area (if given)

    Returns columns listed at config.CALLS_CLEAN, and a report.
    """
    _check_columns(raw, list(RAW_CALL_COLUMNS), "Calls for service file")
    rows_in = len(raw)
    dropped: dict[str, int] = {}

    df = raw[list(RAW_CALL_COLUMNS)].rename(columns=RAW_CALL_COLUMNS)

    # 1. Date
    df["received_date"] = _parse_day(df["received_date"].astype(str))
    df = _drop(df, df["received_date"].isna(), "received date missing or unreadable", dropped)
    start, end = pd.Timestamp(config.DATA_START), pd.Timestamp(config.DATA_END)
    df = _drop(df, (df["received_date"] < start) | (df["received_date"] > end),
               f"received outside {config.DATA_START} to {config.DATA_END}", dropped)

    # 2. Hour
    hour_raw = pd.to_numeric(df["received_hour"], errors="coerce")
    bad_hour = hour_raw.isna() | (hour_raw % 100 != 0) | ~hour_raw.between(0, 2300)
    df = _drop(df, bad_hour, "received hour missing or not a whole hour", dropped)
    df["hour"] = (pd.to_numeric(df["received_hour"]) // 100).astype(int)

    # 3. Priority and who started the call
    df["priority"] = pd.to_numeric(df["priority"], errors="coerce")
    df = _drop(df, ~df["priority"].between(1, 7), "priority missing or not 1-7", dropped)
    df["priority"] = df["priority"].astype(int)
    who = df["initiated_by"].astype("string").str.strip().str.upper().map({"C": "citizen", "O": "officer"})
    df = _drop(df, who.isna(), "initiated by missing or not C/O", dropped)
    df["initiated_by"] = who.loc[df.index].astype(str)

    # 4. Location
    df = _drop(df, df["x"].isna() | df["y"].isna(), "no location (x/y missing)", dropped)
    calls = gpd.GeoDataFrame(
        df.drop(columns=["received_hour", "x", "y"]),
        geometry=gpd.points_from_xy(df["x"], df["y"]),
        crs=config.CRS_CALLS_RAW,
    ).to_crs(config.CRS_METRES)
    calls = _drop(calls, _outside_ottawa(calls.geometry.x, calls.geometry.y), "location outside Ottawa", dropped)
    calls = _drop_outside_study_area(calls, study_area, dropped)

    # 5. Flags
    calls["is_night"] = calls["hour"].isin(config.CALLS_NIGHT_HOURS)
    calls["is_urgent"] = calls["priority"] <= config.URGENT_PRIORITY_MAX

    columns = ["call_id", "received_date", "hour", "priority", "initiated_by", "is_night",
               "is_urgent", "neighbourhood", "intersection", "geometry"]
    calls = calls[columns].reset_index(drop=True)

    night = calls["is_night"]
    notes = {
        "started by a citizen": int((calls["initiated_by"] == "citizen").sum()),
        "started by an officer": int((calls["initiated_by"] == "officer").sum()),
        "night-time calls started by a citizen": int((night & (calls["initiated_by"] == "citizen")).sum()),
        "night-time urgent calls (priority 1-2)": int((night & calls["is_urgent"]).sum()),
        "no intersection name (location still present)": int(calls["intersection"].isna().sum()),
    }
    report = CleaningReport("calls for service", rows_in, len(calls), dropped, notes)
    assert report.rows_in - sum(dropped.values()) == report.rows_out
    return calls, report


# ---------------------------------------------------------------------------
# Bars and pubs (from OpenStreetMap)
# ---------------------------------------------------------------------------
def load_raw_venues(path: Path = config.RAW_OSM_VENUES) -> gpd.GeoDataFrame:
    """Read the bars/pubs file made by `python scripts/00_download_osm.py --venues`."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found in {path.parent}. "
            "Run: python scripts/00_download_osm.py --venues   (needs internet)"
        )
    raw = gpd.read_parquet(path)
    _check_columns(raw, ["venue_id", "kind", "name", "opening_hours", "geometry"], "Venues file")
    return raw


def clean_venues(
    raw: gpd.GeoDataFrame, study_area: BaseGeometry | None = None
) -> tuple[gpd.GeoDataFrame, CleaningReport]:
    """Clean the bars, pubs and nightclubs from OpenStreetMap.

    Steps:
        - drop venues with no location
        - drop exact repeats of the same OpenStreetMap feature
        - convert to config.CRS_METRES and drop venues outside Ottawa, then
          outside study_area (if given)

    opening_hours is kept as OpenStreetMap writes it (e.g. "Mo-Su 11:00-02:00"),
    often missing; it isn't parsed yet.

    Returns columns: venue_id, kind, name, opening_hours, geometry.
    """
    rows_in = len(raw)
    dropped: dict[str, int] = {}

    venues = raw[["venue_id", "kind", "name", "opening_hours", "geometry"]].copy()
    venues = _drop(venues, venues.geometry.isna() | venues.geometry.is_empty, "no location", dropped)
    venues = _drop(venues, venues["venue_id"].duplicated(), "same OpenStreetMap feature listed twice", dropped)

    venues = venues.to_crs(config.CRS_METRES)
    venues = _drop(venues, _outside_ottawa(venues.geometry.x, venues.geometry.y), "location outside Ottawa", dropped)
    venues = _drop_outside_study_area(venues, study_area, dropped)
    venues = gpd.GeoDataFrame(venues.reset_index(drop=True), geometry="geometry", crs=config.CRS_METRES)

    notes = {f"kind = {k}": int(n) for k, n in venues["kind"].value_counts().items()}
    notes["with opening hours"] = int(venues["opening_hours"].notna().sum())
    report = CleaningReport("bars and pubs", rows_in, len(venues), dropped, notes)
    assert report.rows_in - sum(dropped.values()) == report.rows_out
    return venues, report


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_parquet(table: pd.DataFrame | gpd.GeoDataFrame, path: Path) -> None:
    """Save a table as Parquet, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)


def save_reports(reports: list[CleaningReport], path: Path = config.CLEANING_REPORT) -> None:
    """Save all cleaning reports to one CSV (for the report's Dataset section)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([r.to_frame() for r in reports], ignore_index=True).to_csv(path, index=False)


def run() -> list[CleaningReport]:
    """Run all of Phase 1: load, clean and save every dataset.

    The study area (made by scripts/00_download_osm.py) is required: every
    table is cut to it, so it matches the map. Crime and street lights are
    required too. Bus stops, calls for service and
    bars/pubs are skipped with a warning if their raw file is missing, so a
    teammate without them can still run the rest.

    Returns the cleaning reports so the script can print them.
    """
    study_area = load_study_area()

    crime, crime_report = clean_crime(load_raw_crime(), study_area)
    save_parquet(crime, config.CRIME_CLEAN)

    lights, lights_report = clean_lights(load_raw_lights(), study_area)
    save_parquet(lights, config.LIGHTS_CLEAN)
    reports = [crime_report, lights_report]

    optional = [
        ("bus stops", load_raw_stops, clean_stops, config.STOPS_CLEAN),
        ("calls for service", load_raw_calls, clean_calls, config.CALLS_CLEAN),
        ("bars and pubs", load_raw_venues, clean_venues, config.VENUES_CLEAN),
    ]
    for name, load, clean, out_path in optional:
        try:
            raw = load()
        except FileNotFoundError as error:
            warnings.warn(f"Skipping {name}: {error}", stacklevel=2)
            continue
        table, report = clean(raw, study_area)
        save_parquet(table, out_path)
        reports.append(report)

    save_reports(reports)
    return reports
