"""Download the OpenStreetMap data for the project (run ONCE, needs internet).

Run:  python scripts/00_download_osm.py               # walking network only
      python scripts/00_download_osm.py --areas       # also water, parks, parking, industrial

Makes: data/raw/osm_walk_raw.graphml   (walking network, longitude/latitude)
       data/raw/osm_areas.parquet      (only with --areas)

The area and tags come from config.py (OSM_BBOX_LATLON, OSM_EXTRA_WAY_TAGS,
OSM_AREA_TAGS). The files are saved as downloaded, not yet in metres:
Phase 2 (02_build_graph.py) converts them. Share the saved files with the
team (e.g. on Google Drive) so nobody has to download again.

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

from saferoutes import config


def download_walk_network(
    bbox: tuple[float, float, float, float] = config.OSM_BBOX_LATLON,
    extra_tags: tuple[str, ...] = config.OSM_EXTRA_WAY_TAGS,
) -> nx.MultiDiGraph:
    """Download the walking network inside bbox (west, south, east, north).

    Extra tags must be registered BEFORE the download, otherwise OSMnx drops them.
    """
    for tag in extra_tags:
        if tag not in ox.settings.useful_tags_way:
            ox.settings.useful_tags_way.append(tag)
    return ox.graph_from_bbox(bbox, network_type="walk")


def download_areas(
    bbox: tuple[float, float, float, float] = config.OSM_BBOX_LATLON,
    tags: dict[str, bool | str | list[str]] = config.OSM_AREA_TAGS,
) -> gpd.GeoDataFrame:
    """Download water, parks, parking lots and industrial land inside bbox.

    Returns columns: kind ("water", "park", "parking", "industrial"), geometry.
    """
    features = ox.features_from_bbox(bbox, tags=tags)

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--areas", action="store_true",
                        help="also download water, parks, parking and industrial areas")
    args = parser.parse_args()

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"Area (west, south, east, north): {config.OSM_BBOX_LATLON}")

    start = time.time()
    print("Downloading the walking network... (a few minutes)")
    G = download_walk_network()
    ox.save_graphml(G, config.RAW_OSM_GRAPH)
    print(f"  {G.number_of_nodes():,} intersections, {G.number_of_edges():,} street segments "
          f"in {time.time() - start:.0f} s")
    print(f"  Saved {config.RAW_OSM_GRAPH} ({_size_mb(config.RAW_OSM_GRAPH):.0f} MB)")

    if args.areas:
        start = time.time()
        print("Downloading water, parks, parking and industrial areas...")
        areas = download_areas()
        areas.to_parquet(config.RAW_OSM_AREAS, index=False)
        print(f"  {len(areas):,} shapes in {time.time() - start:.0f} s: "
              f"{areas['kind'].value_counts().to_dict()}")
        print(f"  Saved {config.RAW_OSM_AREAS}")


def _size_mb(path: Path) -> float:
    return path.stat().st_size / 1_000_000


if __name__ == "__main__":
    main()
