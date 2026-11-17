"""Phase 1: load and clean the raw files.

Run:  python scripts/01_prepare_data.py
Needs: data/raw/ files (see data/README.md)
Makes: data/processed/crime_clean.parquet, lights_clean.parquet
"""

from saferoutes import data_pipeline


def main() -> None:
    reports = data_pipeline.run()
    for report in reports:
        print(report)


if __name__ == "__main__":
    main()
