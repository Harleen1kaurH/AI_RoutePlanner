"""Phase 6: compare the three routes on 100 random night trips.

Run:  python scripts/04_evaluate_routes.py
Needs: outputs of 02_build_graph.py and 03_train_model.py
Makes: results/route_results.csv, results/route_summary.csv,
       results/astar_vs_dijkstra.csv
"""

from saferoutes import config, evaluation


def main() -> None:
    # 1. Load graph (crime numbers built with config.TEST_CUTOFF)
    # 2. Attach ML risk from node_risk_eval (routing.attach_model_risk)
    # 3. Sample trips, get test-period crimes
    # 4. evaluate_trips -> summarize -> save CSVs
    # 5. check_astar_matches_dijkstra -> save CSV, warn if any mismatch
    raise NotImplementedError


if __name__ == "__main__":
    main()
