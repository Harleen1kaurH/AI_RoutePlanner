"""Phase 3: build the training table, train and score the models,
and save risk per intersection.

Run:  python scripts/03_train_model.py
Needs: output of 02_build_graph.py
Makes: data/processed/training_table.parquet, risk_model.joblib,
       node_risk_eval.parquet, node_risk_demo.parquet,
       results/experiment_log.csv
"""

from saferoutes import config, risk_model


def main() -> None:
    # 1. Load graph, lights and monthly counts
    # 2. Build static features + training table for all cut-offs
    #    (risk_model.make_cutoffs, build_static_node_features, build_training_table)
    # 3. Split into train / validation / test (risk_model.split_by_cutoff)
    # 4. Score the naive baseline, train logistic regression and boosting
    # 5. Compare on VALIDATION, run health checks, calibrate, log each run
    # 6. Final test ONCE on config.TEST_CUTOFF, with confidence ranges
    # 7. Save node_risk_eval (TEST_CUTOFF) and node_risk_demo (DEMO_CUTOFF)
    raise NotImplementedError


if __name__ == "__main__":
    main()
