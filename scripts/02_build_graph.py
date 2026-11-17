"""Phase 2: build the street graph and put safety numbers on every street.

Run:  python scripts/02_build_graph.py
Needs: output of 01_prepare_data.py
Makes: data/processed/street_graph.graphml, crime_snapped.parquet,
       crime_counts_monthly.parquet
"""

from saferoutes import street_graph


def main() -> None:
    G = street_graph.run()
    print(f"Graph: {G.number_of_nodes()} intersections, {G.number_of_edges()} streets")


if __name__ == "__main__":
    main()
