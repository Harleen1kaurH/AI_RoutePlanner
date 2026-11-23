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


def test_clean_calls() -> None:
    # Web Mercator x/y near downtown Ottawa
    base = {"Received Date": "1/4/2021 5:00:00 AM", "Neighbourhood": "Centretown",
            "Intersection": "ELGIN ST, LAURIER AVE W", "x": -8425000.0, "y": 5686500.0}
    raw = pd.DataFrame([
        {**base, "OBJECTID": 1, "Received Hour": 2300, "Priority": 2, "Initiated By": "C"},  # night, urgent, citizen
        {**base, "OBJECTID": 2, "Received Hour": 0, "Priority": 4, "Initiated By": "O"},     # midnight counts as night
        {**base, "OBJECTID": 3, "Received Hour": 1400, "Priority": 3, "Initiated By": "C"},  # daytime
        {**base, "OBJECTID": 4, "Received Hour": 1400, "Priority": 3, "Initiated By": "C", "x": None},  # no location
        {**base, "OBJECTID": 5, "Received Hour": 1400, "Priority": 9, "Initiated By": "C"},  # bad priority
    ])
    calls, report = data_pipeline.clean_calls(raw)
    by_id = calls.set_index("call_id")

    assert list(by_id.index) == [1, 2, 3]
    assert by_id.loc[1, "is_night"] and by_id.loc[1, "is_urgent"] and by_id.loc[1, "initiated_by"] == "citizen"
    assert by_id.loc[2, "is_night"] and by_id.loc[2, "hour"] == 0 and by_id.loc[2, "initiated_by"] == "officer"
    assert not by_id.loc[3, "is_night"] and not by_id.loc[3, "is_urgent"]
    assert by_id.loc[1, "received_date"] == pd.Timestamp("2021-01-04")
    assert calls.crs.to_string() == config.CRS_METRES
    assert report.dropped == {"priority missing or not 1-7": 1, "no location (x/y missing)": 1}


def test_clean_venues() -> None:
    import geopandas as gpd
    from shapely.geometry import Point

    raw = gpd.GeoDataFrame(
        {"venue_id": ["node/1", "way/2", "node/1", "node/3"],
         "kind": ["pub", "bar", "pub", "nightclub"],
         "name": ["Heart & Crown", None, "Heart & Crown", "Far Away"],
         "opening_hours": ["Mo-Su 11:00-02:00", None, "Mo-Su 11:00-02:00", None]},
        geometry=[Point(-75.6935, 45.4290), Point(-75.69, 45.42), Point(-75.6935, 45.4290), Point(-60.0, 45.0)],
        crs="EPSG:4326",
    )
    venues, report = data_pipeline.clean_venues(raw)
    assert list(venues["venue_id"]) == ["node/1", "way/2"]
    assert venues.crs.to_string() == config.CRS_METRES
    assert report.dropped == {"same OpenStreetMap feature listed twice": 1, "location outside Ottawa": 1}
    assert report.notes["with opening hours"] == 1


# ---------------------------------------------------------------------------
# Study area
# ---------------------------------------------------------------------------
def _fake_city() -> "gpd.GeoDataFrame":
    """A fake 'City' covering only the south half of the study box (in lat/lon),
    like the river border cutting the box in two."""
    import geopandas as gpd
    from shapely.geometry import box

    west, south, east, north = config.STUDY_BBOX_LATLON
    middle = (south + north) / 2
    return gpd.GeoDataFrame(geometry=[box(west - 1, south - 1, east + 1, middle)], crs="EPSG:4326")


def test_make_study_area_cuts_to_city_and_adds_margin() -> None:
    import geopandas as gpd
    from shapely.geometry import Point

    area = data_pipeline.make_study_area(_fake_city())
    assert area.crs.to_string() == config.CRS_METRES and len(area) == 1
    shape = area.geometry.iloc[0]

    west, south, east, north = config.STUDY_BBOX_LATLON
    middle = (south + north) / 2
    points = gpd.GeoSeries(
        [Point((west + east) / 2, south + 0.01),     # inside box, inside city -> in
         Point((west + east) / 2, north - 0.01),     # inside box, outside city (the "Gatineau" half) -> out
         Point((west + east) / 2, middle + 0.01)],   # just over the border -> out
        crs="EPSG:4326",
    ).to_crs(config.CRS_METRES)
    assert [shape.contains(p) for p in points] == [True, False, False]

    # the margin: a point 500 m south of the box edge is inside, 1,500 m is not
    edge = gpd.GeoSeries([Point((west + east) / 2, south)], crs="EPSG:4326").to_crs(config.CRS_METRES).iloc[0]
    assert shape.contains(Point(edge.x, edge.y - 500))
    assert not shape.contains(Point(edge.x, edge.y - 1_500))


def test_make_study_area_no_overlap_is_an_error() -> None:
    import geopandas as gpd
    from shapely.geometry import box

    far_city = gpd.GeoDataFrame(geometry=[box(-60, 40, -59, 41)], crs="EPSG:4326")
    with pytest.raises(ValueError):
        data_pipeline.make_study_area(far_city)


def test_clean_crime_drops_outside_study_area(raw_crime: pd.DataFrame) -> None:
    from shapely.geometry import Point

    # a study area of 1 km around downtown; crime 7 is in Ottawa but 10 km away
    study_area = Point(DOWNTOWN).buffer(1_000)
    raw = pd.concat([raw_crime, pd.DataFrame([_crime_row(OBJECTID=7, x=DOWNTOWN[0] + 10_000)])])
    clean, report = data_pipeline.clean_crime(raw, study_area)
    assert set(clean["offence_id"]) == {1, 2, 3}
    assert report.dropped["outside the study area"] == 1
    assert report.dropped["location outside Ottawa"] == 1   # still its own reason
