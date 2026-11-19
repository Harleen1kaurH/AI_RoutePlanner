"""Tests for data_pipeline.py on a few hand-made rows.

Run:  pytest
"""

from __future__ import annotations

import pandas as pd
import pytest

from saferoutes import config, data_pipeline

DOWNTOWN = (367966.7, 5031560.6)   # Elgin & Laurier, MTM zone 9


def _crime_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "OBJECTID": 1,
        "Occurred Date": "7/14/2025 5:00:00 AM",
        "Occurred Hour": 2300,
        "Reported Date": "7/15/2025 5:00:00 AM",
        "Offence Summary": "Crimes Against The Person (1000)",
        "Offence Category": "Assaults",
        "Neighbourhood Name": "Centretown",
        "Intersection": "ELGIN ST, LAURIER AVE W",
        "x": DOWNTOWN[0],
        "y": DOWNTOWN[1],
    }
    row.update(overrides)
    return row


@pytest.fixture
def raw_crime() -> pd.DataFrame:
    return pd.DataFrame([
        _crime_row(OBJECTID=1),                                    # night assault -> label
        _crime_row(OBJECTID=2, **{"Occurred Hour": 1400,           # daytime property crime
                                  "Offence Summary": "Crimes Against Property (2000)",
                                  "Offence Category": "Theft $5000 and Under"}),
        _crime_row(OBJECTID=3, **{"Occurred Hour": 0}),            # hour 0 -> unknown
        _crime_row(OBJECTID=4, x=None, y=None),                    # no location -> dropped
        _crime_row(OBJECTID=5, **{"Occurred Date": "3/1/2015 5:00:00 AM"}),  # before 2018 -> dropped
        _crime_row(OBJECTID=6, x=0.0, y=0.0),                      # outside Ottawa -> dropped
    ])


def test_clean_crime_flags(raw_crime: pd.DataFrame) -> None:
    clean, _ = data_pipeline.clean_crime(raw_crime)
    by_id = clean.set_index("offence_id")

    assert by_id.loc[1, "is_person"] and by_id.loc[1, "is_night"]
    assert by_id.loc[1, "hour"] == 23
    assert by_id.loc[1, "occurred_date"] == pd.Timestamp("2025-07-14")   # occurred, not reported

    assert not by_id.loc[2, "is_person"] and not by_id.loc[2, "is_night"]
    assert by_id.loc[2, "offence_group"] == "theft"

    assert by_id.loc[3, "hour_unknown"] and not by_id.loc[3, "is_night"]


def test_clean_crime_drops_and_report(raw_crime: pd.DataFrame) -> None:
    clean, report = data_pipeline.clean_crime(raw_crime)
    assert set(clean["offence_id"]) == {1, 2, 3}
    assert report.rows_in == 6 and report.rows_out == 3
    assert sum(report.dropped.values()) == 3
    assert report.dropped["no location (x/y missing)"] == 1
    assert report.dropped["location outside Ottawa"] == 1


def test_clean_crime_output_is_in_metres(raw_crime: pd.DataFrame) -> None:
    clean, _ = data_pipeline.clean_crime(raw_crime)
    assert clean.crs.to_string() == config.CRS_METRES
    assert clean.geometry.iloc[0].x == pytest.approx(DOWNTOWN[0])


def test_clean_lights_converts_and_drops() -> None:
    raw = pd.DataFrame({
        "OBJECTID": [1, 2, 3],
        "LIGHT_TYPE": ["SL", "PL", "SL"],
        "LIGHTS_NUM": [1, 2, 1],
        "WATTAGE": [150, 70, 150],
        "X": [-8_420_000.0, None, -8_420_000.0],   # Web Mercator, downtown-ish
        "Y": [5_686_000.0, None, 355_890.0],       # last one is a bad position
    })
    lights, report = data_pipeline.clean_lights(raw)
    assert list(lights["light_id"]) == [1]
    assert lights.crs.to_string() == config.CRS_METRES
    x, y = lights.geometry.iloc[0].x, lights.geometry.iloc[0].y
    min_x, min_y, max_x, max_y = config.OTTAWA_BOUNDS_M
    assert min_x <= x <= max_x and min_y <= y <= max_y
    assert report.dropped == {"no location (X/Y missing)": 1, "location outside Ottawa": 1}


def test_clean_stops() -> None:
    raw = pd.DataFrame({
        "X": [0.0] * 4, "Y": [0.0] * 4,                     # not used: lat/lon are
        "FID": [1, 2, 3, 4],
        "F560": ["8767", "0", "0123", "9999"],
        "Location": ["SUSSEX / RIDEAU FALLS", "ELGIN / PARK", "BANK / SOMERSET", "FAR AWAY"],
        "Latitude": [45.440453, 45.413314, None, 10.0],     # 3rd: no location, 4th: outside Ottawa
        "Longitude": [-75.695438, -75.686321, -75.69, -75.7],
        "Shelter": ["Y", " ", "Y", " "],
        "Bench": [" ", "Y", " ", " "],
    })
    stops, report = data_pipeline.clean_stops(raw)
    by_id = stops.set_index("stop_id")

    assert list(by_id.index) == [1, 2]
    assert by_id.loc[1, "stop_code"] == "8767"
    assert pd.isna(by_id.loc[2, "stop_code"])             # "0" means no code
    assert bool(by_id.loc[1, "has_shelter"]) and not bool(by_id.loc[1, "has_bench"])
    assert not bool(by_id.loc[2, "has_shelter"]) and bool(by_id.loc[2, "has_bench"])
    assert stops.crs.to_string() == config.CRS_METRES
    min_x, min_y, max_x, max_y = config.OTTAWA_BOUNDS_M
    assert all(min_x <= p.x <= max_x and min_y <= p.y <= max_y for p in stops.geometry)
    assert report.dropped == {"no location (Latitude/Longitude missing)": 1, "location outside Ottawa": 1}
