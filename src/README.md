# `src/`: the project's code

Everything in here is one Python package, **`saferoutes`**. Each file is one phase of the project. The files only **define** functions; the scripts in `scripts/` (and the demo in `app/`) **call** them.

Any script, test or the app can use the code with:

```python
from saferoutes import config, data_pipeline, street_graph, risk_model, routing, evaluation
```

This works after the one-time `pip install -e .` from the project folder (see the main README).

---

## 1. How the files connect

Each file reads the files the previous one saved, and saves files for the next one. No file reaches back into an earlier file's internals; they only share **saved files** and the settings in `config.py`.

```
                        config.py  (settings + file names, read by every file)

 data/raw/  ──►  data_pipeline.py  ──►  street_graph.py  ──►  risk_model.py
 (downloads)      Phase 1: clean        Phase 2: map +        Phase 3: ML
                                         safety numbers        risk model
                                               │                     │
                                               ▼                     ▼
                                          routing.py  ◄──────────────┘
                                          Phase 5: A*, three routes
                                               │
                                ┌──────────────┴──────────────┐
                                ▼                             ▼
                         evaluation.py                 app/streamlit_app.py
                         Phase 6: 100 test trips       Phase 7: demo map
```

| Script | Calls | Phase |
|---|---|---|
| `scripts/00_download_osm.py` | (its own code: OSMnx) | Download the map, bars and pubs (once) |
| `scripts/01_prepare_data.py` | `data_pipeline.run()` | 1 |
| `scripts/02_build_graph.py` | `street_graph.run()` | 2 |
| `scripts/03_train_model.py` | functions in `risk_model` | 3 |
| `scripts/04_evaluate_routes.py` | functions in `evaluation` (which uses `routing`) | 6 |
| `app/streamlit_app.py` | functions in `routing` | 7 |

---

## 2. The files one by one

Status: ✅ done · 🟡 stub (function names, inputs and outputs written; body still `raise NotImplementedError`)

### `config.py`: shared settings ✅

**What it does:** holds every decision from the project plan in one place: file paths, coordinate systems, night hours, cut-off dates, distances, weights, the study area. Contains no functions.

**Input:** nothing. **Output:** values like `config.NIGHT_HOURS` or `config.CRIME_CLEAN`, imported by every other file.

**Rule:** never type a setting or file path directly in your code; add it here and use `config.NAME`. Agree with the team before changing a value.

---

### `data_pipeline.py`: Phase 1, load and clean ✅

**Owner:** Data person

**What it does:** reads each raw download, keeps the columns we need, fixes formats (dates, hours, yes/no flags), drops bad rows while counting why, converts every location to metres (EPSG:2951), and saves clean tables.

**Input** (`data/raw/`):

| Raw file | Required? |
|---|---|
| `Criminal_Offences_Open_Data*.csv` (Ottawa Police) | Yes |
| `Street_Lights*.csv` (Open Ottawa) | Yes |
| `OC_Transpo_Stops*.csv` | Optional |
| `Calls_For_Service*.csv` (Ottawa Police) | Optional |
| `osm_venues.parquet` (from `00_download_osm.py --venues`) | Optional |

**Output** (`data/processed/` and `results/`):

| File | One row per | Main columns |
|---|---|---|
| `crime_clean.parquet` | crime | occurred_date, hour, offence_group, is_person, is_night, hour_unknown, geometry |
| `lights_clean.parquet` | light record | light_type, n_heads, wattage, geometry |
| `stops_clean.parquet` | bus stop | stop_code, stop_name, has_shelter, has_bench, geometry |
| `calls_clean.parquet` | call for service | received_date, hour, priority, initiated_by, is_night, is_urgent, geometry |
| `venues_clean.parquet` | bar / pub / nightclub | kind, name, opening_hours, geometry |
| `results/cleaning_report.csv` | drop reason or note | dataset, kind, item, count |

**Main functions:** for each dataset, a `load_raw_…` (read the file as downloaded) and a `clean_…` (returns the clean table **and** a `CleaningReport`). `run()` does all of them in order and saves the results. Helpers starting with `_` are for use inside this file only.

**Used by:** every later phase.

---

### `street_graph.py`: Phase 2, map and safety numbers 🟡

**Owner:** Data person

**What it does:** loads the OpenStreetMap walking network, converts it to metres, keeps urban Ottawa only (removing Gatineau), attaches every crime and call to its nearest intersection, counts them per intersection per month, and puts safety numbers on every street. **No machine learning.**

**Input:** `data/raw/osm_walk_raw.graphml` (from `00_download_osm.py`), `crime_clean`, `lights_clean`, `calls_clean`.

**Output** (`data/processed/`):

| File | Contents |
|---|---|
| `street_graph.graphml` | Intersections (nodes with x, y) and streets (edges). Every street carries: length, lights_per_100m, longest_dark_gap_m, darkness, street_type, isolated, underpass, past_crimes_12m, crime_score |
| `crime_snapped.parquet` | `crime_clean` + `node_id` and `snap_distance_m` |
| `crime_counts_monthly.parquet` | node_id, month, n_person_night, n_all, n_property, n_mischief, n_theft |
| `call_counts_monthly.parquet` | node_id, month, n_calls_night_citizen, n_calls_night_urgent, n_calls_all |

**Main functions:** `load_osm_walk_graph` → `restrict_to_urban_area` → `snap_to_nodes` (crimes and calls) → `monthly_crime_counts`, `monthly_call_counts` → `add_light_features`, `add_street_type_features`, `add_crime_count_features` → `save_graph`. `load_graph` reads the saved graph back for later phases.

**Important:** the crime numbers on streets count only crimes **before a cut-off date** (Oct 1, 2025 by default), so the evaluation never sees the crimes it's tested on.

**Used by:** `risk_model`, `routing`, `evaluation`, the app.

---

### `risk_model.py`: Phase 3, the ML model 🟡

**Owner:** ML person

**What it does:** for every intersection, predicts the chance of at least one night-time crime against the person in the next 3 months. It builds the training table (features from **before** each cut-off date, label from **after** it), trains and compares three models (simple count, logistic regression, gradient boosting), runs health checks, calibrates the probabilities, and saves a risk number for every intersection.

**Input:** `street_graph.graphml`, `crime_counts_monthly`, `call_counts_monthly`, `lights_clean`, `stops_clean`, `venues_clean`.

**Output:**

| File | Contents |
|---|---|
| `data/processed/training_table.parquet` | node_id, cutoff, feature columns, label (0/1), n_future |
| `data/processed/risk_model.joblib` | The trained model |
| `data/processed/node_risk_eval.parquet` | node_id, probability, risk_score at the **test** cut-off (Oct 1, 2025), for the evaluation |
| `data/processed/node_risk_demo.parquet` | Same, at the **demo** cut-off (Jan 1, 2026) |
| `results/experiment_log.csv` | One row per experiment: features, settings, scores |

**Main functions:**
- Building the table: `make_cutoffs`, `build_static_node_features`, `build_time_features`, `build_labels`, `build_training_table`, `split_by_cutoff`
- Models: `naive_scores`, `train_logistic`, `train_boosting`, `predict_proba`
- Scores: `evaluate_scores` (AUC, PR-AUC, PAI@5%, Brier), `pai_at`, `bootstrap_ci`
- Checks: `shuffled_label_auc` (leakage), `ablation` (which feature groups matter), `log_run`
- Output: `calibrate`, `predict_node_risk`, `save_model`, `load_model`

The feature names, grouped for the ablation check, are in `FEATURE_GROUPS` at the top of the file.

**Used by:** `routing` (Route 3), `evaluation`, the app.

---

### `routing.py`: Phase 5, A* and the three routes 🟡

**Owner:** Routing person

**What it does:** turns each street's numbers into a **cost** and uses A* to find the cheapest route. The three routes differ only in the cost:

| Route | `risk_source` | Cost of a street |
|---|---|---|
| 1. Shortest | `"none"` | length |
| 2. Safer, rules | `"counts"` | length × (1 + w_crime·crime_score + w_dark·darkness + w_isolated·isolated) |
| 3. Safer, ML | `"model"` | same, with `model_risk_score` instead of `crime_score` |

**Input:** the graph from Phase 2, `node_risk_*.parquet` from Phase 3, a start and end intersection, and `Weights`.

**Output:** `Route` objects in memory (list of intersections, length, cost, number of intersections A* explored). Nothing is saved to disk.

**Main functions:** `attach_model_risk` (puts the ML risk on streets), `nearest_node`, `edge_cost`, `straight_line_heuristic`, `astar` (our own implementation), `dijkstra_cost` (correctness check), `shortest_route`, `safer_route_counts`, `safer_route_model`, `three_routes`, and optional `pareto_routes`.

**Checked by:** `tests/test_routing.py` (A* must give the same cost as Dijkstra).

**Used by:** `evaluation`, the app.

---

### `evaluation.py`: Phase 6, testing the routes 🟡

**Owner:** Data person

**What it does:** picks 100 random night trips, builds all three routes for each (using only data from before Oct 1, 2025), and measures each route against **real crimes from Oct–Dec 2025** that the system never saw. Also checks A* against Dijkstra on every trip.

**Input:** the graph (with ML risk attached), `crime_clean` (for the test-period crimes).

**Output** (saved by `scripts/04_evaluate_routes.py`):

| File | Contents |
|---|---|
| `results/route_results.csv` | One row per (trip, route): length_m, extra_pct, crimes_passed, pct_lit, isolated_m |
| `results/route_summary.csv` | Averages per route type: the report's main results table |
| `results/astar_vs_dijkstra.csv` | A* cost vs Dijkstra cost for every trip and route |

**Main functions:** `sample_trips`, `crimes_in_test_period`, `route_measures`, `evaluate_trips`, `summarize`, `check_astar_matches_dijkstra`.

---

## 3. Hand-off files at a glance

| File | Made by | Read by |
|---|---|---|
| `crime_clean.parquet` | `data_pipeline` | `street_graph`, `evaluation` |
| `lights_clean.parquet` | `data_pipeline` | `street_graph`, `risk_model` |
| `stops_clean.parquet` | `data_pipeline` | `risk_model` |
| `calls_clean.parquet` | `data_pipeline` | `street_graph` |
| `venues_clean.parquet` | `data_pipeline` | `risk_model` |
| `street_graph.graphml` | `street_graph` | `risk_model`, `routing`, `evaluation`, app |
| `crime_snapped.parquet` | `street_graph` | the app (to show past incidents on the map); available to anyone needing each crime's intersection |
| `crime_counts_monthly.parquet` | `street_graph` | `risk_model`, `street_graph` (crime numbers on streets) |
| `call_counts_monthly.parquet` | `street_graph` | `risk_model` |
| `node_risk_eval.parquet` | `risk_model` | `routing` (via `evaluation`) |
| `node_risk_demo.parquet` | `risk_model` | `routing` (via the app) |

Exact column names and types are listed next to each path in `config.py`. **If you change a column, update `config.py`, this table and the main README, and tell the team.**

---

## 4. Conventions for everyone

- **Type hints on every function**, for what goes in and what comes out (e.g. `-> tuple[gpd.GeoDataFrame, CleaningReport]`).
- **A docstring on every function**: what it does, and anything surprising.
- **Names starting with `_`** are helpers for use inside that file only.
- **Locations are always in metres (EPSG:2951)** once Phase 1 has run.
- **Use `table["intersection"]`, not `table.intersection`.** Geopandas has a built-in function with that name.
- **Stubs say `raise NotImplementedError`.** Replace the body; keep the name, inputs and outputs unless the team agrees to change them.
- **Add a test** in `tests/` for anything with a rule you can check by hand on a few rows.
