"""Phase 1: load and clean the raw files.

Run:  python scripts/01_prepare_data.py
Needs: data/raw/ files (see data/README.md)
Makes: data/processed/crime_clean.parquet, lights_clean.parquet, stops_clean.parquet,
       results/cleaning_report.csv
"""

from saferoutes import config, data_pipeline


def main() -> None:
    reports = data_pipeline.run()
    for report in reports:
        print(report)
    print(f"\nSaved {config.CRIME_CLEAN}")
    print(f"Saved {config.LIGHTS_CLEAN}")
    print(f"Saved {config.STOPS_CLEAN}")
    print(f"Saved {config.CLEANING_REPORT}")


if __name__ == "__main__":
    main()
