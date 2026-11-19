# Safer Night Walking Routes in Ottawa

COMP 3106 (Introduction to AI) term project, Fall 2026.

A route planner for walking in Ottawa at night. For any start and end point it finds three routes:

1. **Shortest route:** A* on distance only
2. **Safer route (rules):** A* with penalties for past crime counts, darkness and isolated streets
3. **Safer route (ML):** the same formula, but the crime number comes from a trained risk model

We then test which routes avoid real crimes that happened later (Oct–Dec 2025), and whether the ML model is worth using over simple counts. The full plan is in the shared project plan doc.

## Who owns what

| Role | Name | Files |
|---|---|---|
| Data & map | _TBD_ | `data_pipeline.py`, `street_graph.py`, `evaluation.py` |
| ML model | _TBD_ | `risk_model.py` |
| Routing & demo | _TBD_ | `routing.py`, `app/streamlit_app.py`, `tests/test_routing.py` |

`config.py` is shared: agree with the team before changing a value.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # makes "import saferoutes" work everywhere
```

Then download the raw data files into `data/raw/` (see `data/README.md`).

## How to run

Run the scripts in order. Each one saves its output, so later steps don't redo earlier ones.

```bash
python scripts/00_download_osm.py --venues   # once: OpenStreetMap map + bars/pubs (needs internet)
python scripts/01_prepare_data.py     # Phase 1: clean crime, lights, stops, calls, bars
python scripts/02_build_graph.py      # Phase 2: street graph + safety numbers
python scripts/03_train_model.py      # Phase 3: ML model + risk per intersection
python scripts/04_evaluate_routes.py  # Phase 6: compare the three routes
streamlit run app/streamlit_app.py    # Phase 7: demo map
pytest                                # tests (A* must match Dijkstra)
```

## Project layout

```
src/saferoutes/
  config.py          shared settings: paths, dates, distances, weights
  data_pipeline.py   Phase 1  load and clean raw data
  street_graph.py    Phase 2  graph, snap crimes, safety numbers on streets
  risk_model.py      Phase 3  training table, models, scoring, node risk
  routing.py         Phase 5  edge cost, heuristic, A*, three routes
  evaluation.py      Phase 6  100 test trips and route measures
scripts/             run the phases in order
app/                 Streamlit demo
tests/               automated checks
notebooks/           exploration only (never imported by src/)
data/raw/            downloaded files (not in Git)
data/processed/      files our code creates (not in Git)
results/             tables and figures for the report (not in Git)
```

## Hand-off files between team members

| File | Made by | Used by | Columns / attributes |
|---|---|---|---|
| `crime_clean.parquet` | Phase 1 | everyone | offence_id, occurred_date, reported_date, hour, offence_summary, offence_category, offence_group, is_person, is_night, hour_unknown, neighbourhood, intersection, geometry |
| `lights_clean.parquet` | Phase 1 | Phases 2, 3 | light_id, light_type, n_heads, wattage, geometry |
| `stops_clean.parquet` | Phase 1 | Phase 3 | stop_id, stop_code, stop_name, has_shelter, has_bench, geometry |
| `calls_clean.parquet` | Phase 1 | Phase 2 | call_id, received_date, hour, priority, initiated_by, is_night, is_urgent, neighbourhood, intersection, geometry |
| `venues_clean.parquet` | Phase 1 | Phase 3 | venue_id, kind, name, opening_hours, geometry |
| `crime_counts_monthly.parquet` | Phase 2 | Phases 3, 6 | node_id, month, n_person_night, n_all, n_property, n_mischief, n_theft |
| `call_counts_monthly.parquet` | Phase 2 | Phase 3 | node_id, month, n_calls_night_citizen, n_calls_night_urgent, n_calls_all |
| `street_graph.graphml` | Phase 2 | Phases 3, 5, 6, 7 | edge attributes in `config.EDGE_ATTRIBUTES` |
| `node_risk_eval.parquet` / `node_risk_demo.parquet` | Phase 3 | Phases 5, 6, 7 | node_id, probability, risk_score |

If you need to change a column name, update `config.py` and this table, and tell the team.

## Working with Git

- Never commit straight to `main`. Make a branch, e.g. `data/loaders`, `ml/baselines`, `routing/astar`.
- Open a pull request, and have one teammate review it before merging.
- Commit small and often (several times a week). Write clear messages, e.g. "Add lights-per-100m edge feature".
- Never commit data files. `.gitignore` already blocks `data/raw/`, `data/processed/` and `results/`.

## Key dates

- **Dec 4:** feature freeze. After this, only bug fixes, final results and writing.
- **Dec 7–11:** demos.
- **Dec 11:** code and report due.
