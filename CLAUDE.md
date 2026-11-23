# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

COMP 3106 team project (Fall 2026): night-time walking route planner for Ottawa. For a start/end point it returns three A* routes — shortest (length only), safer-by-rules (crime counts + darkness + isolation penalties), and safer-by-ML (same formula, crime term from a trained risk model) — then evaluates them against real crimes from a held-out period (Oct–Dec 2025). The full plan is in `Safer Night Walking Routes in Ottawa – Project Plan.pdf`.

Many functions in `street_graph.py`, `risk_model.py`, `routing.py` and `evaluation.py` are still typed stubs raising `NotImplementedError`; their docstrings are the spec. Phase 1 (`data_pipeline.py`) is implemented.

## Commands

```bash
pip install -r requirements.txt && pip install -e .   # Python >= 3.10; editable install makes `import saferoutes` work

python scripts/00_download_osm.py --venues   # once, needs internet: OSM walk graph (+ bars/pubs); --areas, --skip-graph also exist
python scripts/01_prepare_data.py            # Phase 1: clean raw data -> data/processed/*.parquet + results/cleaning_report.csv
python scripts/02_build_graph.py             # Phase 2: street graph + edge safety attributes
python scripts/03_train_model.py             # Phase 3: risk model + node risk
python scripts/04_evaluate_routes.py         # Phase 6: compare the three routes
streamlit run app/streamlit_app.py           # Phase 7: demo map

pytest                                       # all tests (pytest config in pyproject.toml, pythonpath=src)
pytest tests/test_routing.py::test_name      # single test
```

Raw data must be downloaded by hand into `data/raw/` with the exact names in `data/README.md`. `data/raw/`, `data/processed/` and `results/` are git-ignored — never commit data.

## Architecture

- **Pipeline of phases, connected by files.** Each `scripts/0N_*.py` is a thin wrapper around one module in `src/saferoutes/`; each phase writes Parquet/GraphML to `data/processed/` and later phases read those files rather than recomputing. Modules are owned by different team members (see README "Who owns what").
- **`config.py` is the single source of truth and the inter-team contract.** It holds every path, the column schema of each hand-off file (in comments next to the path constant), `EDGE_ATTRIBUTES` every graph edge must carry, CRS constants, dates/cut-offs, distances and routing weights. Never hard-code these values elsewhere. Changing a value or column name is a team decision: update `config.py` *and* the "Hand-off files" table in `README.md`.
- **Coordinates:** everything is converted to `CRS_METRES` (EPSG:2951, MTM zone 9). Raw sources differ: crime is already EPSG:2951, lights and calls are EPSG:3857, bus stops are lat/lon, OSM downloads are lat/lon (projected in Phase 2).
- **Phase 1 (`data_pipeline.py`):** `load_raw_X()` / `clean_X(raw) -> (GeoDataFrame, CleaningReport)` pairs. Every dropped row goes through `_drop(..., reason, dropped)` so `CleaningReport` records counts per reason (used in the written report). Crime and lights are required; stops, calls and venues are skipped with a warning if their raw file is missing. Some raw files are located by glob (download adds numbers to names) via `_find_raw_file`.
- **Phase 2 (`street_graph.py`):** builds the OSMnx walk graph, snaps crimes to the nearest intersection (≤ `SNAP_MAX_DISTANCE_M`), produces monthly per-node crime/call counts and edge attributes. `model_risk_score` is *not* added here — `routing.attach_model_risk` adds it from `node_risk_*.parquet`.
- **Phase 3 (`risk_model.py`):** one row per (intersection, cut-off), with a cut-off on the 1st of every month and a row for *every* urban intersection, not just ones with past crime. Label = at least one night person-crime in the next `LABEL_MONTHS` (3). Features use only data before the row's cut-off. Feature groups are crime history (1/3/12 months plus a decaying count), general activity, neighbours within 250 m, lighting, street form, optional busyness (stops, bars) and season. Street-level facts are summarized at the intersection, because labels only exist there.
  - Splits: training cut-offs Jan 2022 → Apr 2025 (the last labels end Jun 30), validation at Jul 1 2025, test at Oct 1 2025. Training starts in 2022 because each row needs a year of history and 2020–21 were pandemic years. The **test cut-off is used once**, at the end: retrain on train + validation, then predict.
  - Models: the simple-count baseline (night person-crimes in the last 12 months), logistic regression and gradient boosting, all with `class_weight="balanced"`. Never judge by accuracy. Score on validation with AUC, PR-AUC, PAI at 5% and Brier. An ML model is chosen only if it beats the baseline.
  - Health checks: compare training and validation scores (overfitting); retrain on shuffled labels, where AUC should be about 0.5, and treat any AUC above 0.95 as leakage; drop one feature group at a time (`ablation`); log every run to `results/experiment_log.csv`. Calibrate on the validation predictions, because Route 3 uses the probability values themselves, not just their ranking. Test scores get 1,000-resample bootstrap confidence ranges.
  - Output: `risk_score` = probability ÷ city average (1 = average). The eval file uses `TEST_CUTOFF`. The demo file comes from a model retrained on all data, predicting at `DEMO_CUTOFF`.
- **Phase 4: decision gate.** There is no code module for it. Route 3 becomes the demo's main safer route only if (1) on the test quarter the model beats the simple count on AUC and PAI at 5% with non-overlapping confidence ranges and passes calibration, **and** (2) in Phase 6, Route 3 passes fewer real crimes than Route 2 for about the same extra distance. Otherwise Route 2 is the default. All three routes are always built and reported. The rule is fixed in advance and must not be bent after seeing test results.
- **Phase 5 (`routing.py`):** the three routes share one cost function selected by `risk_source` ("none" / "counts" / "model", mapped in `CRIME_ATTRIBUTE`): `cost = length * (1 + w_crime*crime + w_dark*darkness + w_isolated*isolated)`. Routes 2 and 3 differ **only** in the crime ingredient (`crime_score` vs `model_risk_score`) and must use the same weights, which keeps the comparison fair. Cost is always ≥ length, which keeps the straight-line-distance heuristic admissible. `astar` is a hand-written implementation (course requirement); on multi-edges use the cheapest edge. Tests require A* cost to equal networkx Dijkstra (`dijkstra_cost`). Tune the weights (default 0.5 each) so routes are usually at most `MAX_DETOUR_RATIO` (25%) longer than the shortest. `pareto_routes` is optional, and only if Routes 1–3 are done by Nov 27.
- **Phase 6 (`evaluation.py`):** 100 random trips (0.5–3 km apart, `RANDOM_SEED`) using only pre-`TEST_CUTOFF` data, for both the counts and the ML risk. For each route, measure: length and extra %, real Oct–Dec 2025 night person-crimes within 50 m, % of length lit (≥ 3 lights per 100 m), and isolated metres. Run the A* = Dijkstra check on every trip. The results table is Check 2 of the decision gate.
- **Phase 7 (`app/streamlit_app.py`):** a downtown map (`DEMO_AREA`) where you click a start and an end, the three routes drawn with their stats, weight sliders that re-route, and an "add incident" button that raises a street's crime count so A* re-routes live. It is for a 13-minute demo.
- `notebooks/` is exploration only and must never be imported by `src/`.

## Where the plan PDF and the code differ

The code and `config.py` are authoritative over the PDF:
- The node risk files are `node_risk_eval.parquet` / `node_risk_demo.parquet` with a `node_id` column, not the plan's `node_risk.parquet` / `intersection_id`.
- The OSM network is downloaded once by `scripts/00_download_osm.py`, not inside Phase 2.
- The plan counts **underpasses** as `isolated` = 1, but `config.EDGE_ATTRIBUTES` keeps `underpass` as a separate attribute and the cost formula reads only `isolated`. Underpasses must be folded into `isolated`, or they will never be penalized.
- `crime_score` depends on the cut-off (`TEST_CUTOFF` for evaluation, `DEMO_CUTOFF` for the demo), but there is only one `STREET_GRAPH` path. Settle how the two versions are kept apart before implementing Phase 2.

## Report limitations to keep in mind

The written report must state these limitations: crimes are snapped to intersections, so mid-block risk isn't visible; reported crime reflects policing and reporting patterns, not true risk; the model covers night only; and the lighting data covers City-maintained lights only. Ethics: the UI and the report must avoid labelling neighbourhoods as "dangerous".

## Data gotchas

- "Night" means 6 pm–6 am. The label is night-time **crimes against the person** only; other crimes are background features.
- Crime `Occurred Hour` is stored like `2300`; hour `0` (about 4× as common as 1 am) means "time unknown" (`hour_unknown`) and is never counted as night. In calls-for-service, hour 0 is real midnight and *is* night (`CALLS_NIGHT_HOURS`).
- Crimes that occurred before `DATA_START` (2018) are dropped; points outside `OTTAWA_BOUNDS_M` are dropped as location errors.
- Calls for service have no call type, so they are features only, never labels.
- ~6,000 street-light records share positions; `lights_clean` keeps them all, and Phase 2 must de-duplicate by rounded position (`LIGHT_POSITION_ROUND_M`) when computing lighting features.

## Git workflow

Never commit to `main`; use topic branches (e.g. `data/...`, `ml/...`, `routing/...`) and PRs reviewed by a teammate. Feature freeze is Dec 4, 2026; code and report due Dec 11.
