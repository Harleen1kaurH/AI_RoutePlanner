"""Phase 1: load and clean the raw files.

Run:  python scripts/01_prepare_data.py
Needs: data/raw/ files (see data/README.md)
Makes: data/processed/crime_clean.parquet, lights_clean.parquet, stops_clean.parquet,
       calls_clean.parquet, venues_clean.parquet, results/cleaning_report.csv
       (stops, calls and venues are skipped with a warning if their raw file is missing)
"""

from saferoutes import config, data_pipeline


def main() -> None:
    reports = data_pipeline.run()
    for report in reports:
        print(report)
    print()
    for path in (config.CRIME_CLEAN, config.LIGHTS_CLEAN, config.STOPS_CLEAN,
                 config.CALLS_CLEAN, config.VENUES_CLEAN, config.CLEANING_REPORT):
        if path.exists():
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
