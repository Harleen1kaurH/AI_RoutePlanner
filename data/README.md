# Data

Raw files are **not** stored in Git: they're too big and anyone can download them. Put them in `data/raw/` with the exact names below, so `config.py` can find them.

| Save as (in `data/raw/`) | Source | How to get it |
|---|---|---|
| `Criminal_Offences_Open_Data.csv` | Ottawa Police Community Safety Data Portal | https://data.ottawapolice.ca/datasets/0356deb79cfd4488956116fda366df38_0 → Download → CSV. Rename the file (the download adds a number to the name). |
| `Street_Lights.csv` | Open Ottawa | https://open.ottawa.ca/datasets/street-lights/about → Download → CSV |
| `OC_Transpo_Stops.csv` | Open Ottawa / OC Transpo | Bus stop locations with shelter and bench. Keep the downloaded name (any `OC_Transpo_Stops*.csv` is found). |
| `gtfs/` (folder, optional) | OC Transpo | https://octranspo.com/en/plan-your-trip/travel-tools/developers → GTFS static zip → unzip into this folder |
| `agco_liquor_licences.csv` (optional) | AGCO | https://www.agco.ca/en/general/data-inventory → liquor sales licences CSV |

The OpenStreetMap street network is downloaded by `python scripts/00_download_osm.py` (add `--areas` for water, parks, parking and industrial land). It saves `osm_walk_raw.graphml` (and `osm_areas.parquet`) here. Run it once, then share the files with the team. The download area is `OSM_BBOX_LATLON` in `config.py`.

Team members can also share these files through the team's Google Drive folder instead of downloading them again.

## What the code creates

`data/processed/` is filled by the scripts. See the "Hand-off files" table in the main README for each file's columns.

## Notes on the raw data

- Crime locations are snapped by the police to the nearest **intersection** (column `Intersection`, e.g. "ELGIN ST, LAURIER AVE W"), with x/y in MTM zone 9 (EPSG:2951).
- Street light x/y are in Web Mercator (EPSG:3857). The code converts both to EPSG:2951.
- `Occurred Hour` is stored like `2300`. Hour `0` appears far more often than 1 am and is treated as "time unknown".
