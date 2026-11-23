"""Download the OpenStreetMap data for the project (run ONCE, needs internet).

This is step 1 of Phase 1; then run 01_prepare_data.py.

Run:  python scripts/00_download_osm.py --venues --areas   # everything (recommended)
      python scripts/00_download_osm.py                    # study area + walking network only
      python scripts/00_download_osm.py --skip-graph --venues   # only bars (map already downloaded)

Makes: data/raw/study_area.parquet     (the study area, in metres; always made,
                                        or reused with --skip-graph)
       data/raw/osm_walk_raw.graphml   (walking network, longitude/latitude)
       data/raw/osm_areas.parquet      (only with --areas)
       data/raw/osm_venues.parquet     (only with --venues)

The study area = the team's box (config.STUDY_BBOX_LATLON) widened by
config.STUDY_MARGIN_M and cut to the City of Ottawa boundary (removes Gatineau);
see data_pipeline.make_study_area. Everything is downloaded inside it.
Tags come from config.py (OSM_EXTRA_WAY_TAGS, OSM_AREA_TAGS, OSM_VENUE_TAGS).
The map files are saved as downloaded (longitude/latitude): Phase 2 converts
them to metres. Share the saved files with the team (e.g. on Google Drive) so
nobody has to download again.

Needs OSMnx 2 or newer:  pip install "osmnx>=2.0"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd

from shapely.geometry.base import BaseGeometry

from saferoutes import config, data_pipeline


def get_study_area(reuse_saved: bool = False) -> BaseGeometry:
    """Make (or, with reuse_saved, reload) the study area and return it in
    longitude/latitude, the coordinates OSMnx downloads in.

    Making it downloads the City of Ottawa boundary, cuts the widened box to it
    (data_pipeline.make_study_area) and saves the result to config.STUDY_AREA.
    """
    if reuse_saved and config.STUDY_AREA.exists():
        print(f"Using the saved study area {config.STUDY_AREA}")
        area = gpd.read_parquet(config.STUDY_AREA)
    else:
        print(f"Downloading the boundary of {config.CITY_PLACE_NAME}...")
        city = ox.geocode_to_gdf(config.CITY_PLACE_NAME)
        area = data_pipeline.make_study_area(city)
        area.to_parquet(config.STUDY_AREA, index=False)
        print(f"  Saved {config.STUDY_AREA}")
    print(f"  Study area: {area['area_km2'].iloc[0]:,.0f} km² "
          f"(box {config.STUDY_BBOX_LATLON} + {config.STUDY_MARGIN_M:,.0f} m margin, cut to the City)")
    return area.to_crs(config.CRS_LATLON).union_all()


def download_walk_network(
    area: BaseGeometry,
    extra_tags: tuple[str, ...] = config.OSM_EXTRA_WAY_TAGS,
) -> nx.MultiDiGraph:
    """Download the walking network inside `area` (a shape in longitude/latitude).

    Streets with an end outside the area are cut off at the last intersection
    inside it. OSMnx keeps only the largest connected piece.
    Extra tags must be registered BEFORE the download, otherwise OSMnx drops them.
    """
    for tag in extra_tags:
        if tag not in ox.settings.useful_tags_way:
            ox.settings.useful_tags_way.append(tag)
    return ox.graph_from_polygon(area, network_type="walk")


def download_areas(
    area: BaseGeometry,
    tags: dict[str, bool | str | list[str]] = config.OSM_AREA_TAGS,
) -> gpd.GeoDataFrame:
    """Download water, parks, parking lots and industrial land inside `area`.

    Returns columns: kind ("water", "park", "parking", "industrial"), geometry.
    """
    features = ox.features_from_polygon(area, tags=tags)

    kind = pd.Series("other", index=features.index)
    if "landuse" in features:
        kind[features["landuse"] == "industrial"] = "industrial"
    if "amenity" in features:
        kind[features["amenity"] == "parking"] = "parking"
    if "leisure" in features:
        kind[features["leisure"] == "park"] = "park"
    if "natural" in features:
        kind[features["natural"] == "water"] = "water"
    if "waterway" in features:
        kind[features["waterway"].notna()] = "water"

    areas = gpd.GeoDataFrame({"kind": kind.values}, geometry=features.geometry.values, crs=features.crs)
    return areas[areas["kind"] != "other"].reset_index(drop=True)


def download_venues(
    area: BaseGeometry,
    tags: dict[str, bool | str | list[str]] = config.OSM_VENUE_TAGS,
) -> gpd.GeoDataFrame:
    """Download bars, pubs and nightclubs inside `area`.

    Some venues are mapped as a point, others as a building outline; outlines
    are replaced by a point inside them.

    Returns columns: venue_id (e.g. "node/123"), kind, name, opening_hours,
    geometry (point, longitude/latitude).
    """
    features = ox.features_from_polygon(area, tags=tags)

    def column(name: str) -> pd.Series:
        """The tag column `name`, or an empty column if no venue has that tag."""
        return features[name] if name in features else pd.Series(None, index=features.index, dtype=object)

    venue_id = [f"{kind}/{osm_id}" for kind, osm_id in features.index]   # index = (element type, id)
    points = features.geometry.representative_point()
    venues = gpd.GeoDataFrame(
        {
            "venue_id": venue_id,
            "kind": column("amenity").values,
            "name": column("name").values,
            "opening_hours": column("opening_hours").values,
        },
        geometry=points.values,
        crs=features.crs,
    )
    return venues[venues["kind"].isin(tags["amenity"])].reset_index(drop=True)


def main() -> None:
    """Read the command-line options and download the chosen OpenStreetMap data."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--areas", action="store_true",
                        help="also download water, parks, parking and industrial areas")
    parser.add_argument("--venues", action="store_true",
                        help="also download bars, pubs and nightclubs")
    parser.add_argument("--skip-graph", action="store_true",
                        help="don't download the walking network again")
    args = parser.parse_args()

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    area = get_study_area(reuse_saved=args.skip_graph)

    if not args.skip_graph:
        start = time.time()
        print("Downloading the walking network... (10+ minutes for the full study area)")
        G = download_walk_network(area)
        ox.save_graphml(G, config.RAW_OSM_GRAPH)
        print(f"  {G.number_of_nodes():,} intersections, {G.number_of_edges():,} street segments "
              f"in {time.time() - start:.0f} s")
        print(f"  Saved {config.RAW_OSM_GRAPH} ({_size_mb(config.RAW_OSM_GRAPH):.0f} MB)")

    if args.areas:
        start = time.time()
        print("Downloading water, parks, parking and industrial areas...")
        areas = download_areas(area)
        areas.to_parquet(config.RAW_OSM_AREAS, index=False)
        print(f"  {len(areas):,} shapes in {time.time() - start:.0f} s: "
              f"{areas['kind'].value_counts().to_dict()}")
        print(f"  Saved {config.RAW_OSM_AREAS}")

    if args.venues:
        start = time.time()
        print("Downloading bars, pubs and nightclubs...")
        venues = download_venues(area)
        venues.to_parquet(config.RAW_OSM_VENUES, index=False)
        print(f"  {len(venues):,} venues in {time.time() - start:.0f} s: "
              f"{venues['kind'].value_counts().to_dict()}")
        print(f"  Saved {config.RAW_OSM_VENUES}")


def _size_mb(path: Path) -> float:
    """File size in megabytes, for the progress messages."""
    return path.stat().st_size / 1_000_000


if __name__ == "__main__":
    main()
