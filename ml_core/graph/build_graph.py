from neo4j import GraphDatabase
import networkx as nx
from sqlalchemy import create_engine, text


# ============================================================
# DATABASE CONNECTIONS
# ============================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "quant_password")

PG_URL = "postgresql://quant_user:quant_password@localhost:5432/quant_db"

pg_engine = create_engine(PG_URL)


# ============================================================
# REPRESENTATIVE RELATIONSHIPS
# ============================================================
#
# These are intentionally representative/demo relationships.
# In a production system these should come from real sources
# such as SEC disclosures, Wikidata, or licensed datasets.
#

RELATIONSHIPS = [
    ("AAPL", "SUPPLIES", "TSMC"),
    ("NVDA", "SUPPLIES", "TSMC"),
    ("MSFT", "PARTNERS_WITH", "OAI"),
    ("MSFT", "COMPETES_WITH", "AAPL"),
    ("GOOGL", "COMPETES_WITH", "MSFT"),
    ("AMZN", "SUPPLIES", "RIVN"),
    ("SPY", "CONTAINS", "AAPL"),
    ("SPY", "CONTAINS", "MSFT"),
    ("SPY", "CONTAINS", "GOOGL"),
    ("SPY", "CONTAINS", "AMZN"),
    ("SPY", "CONTAINS", "NVDA"),
]


# ============================================================
# POPULATE NEO4J
# ============================================================

def build_neo4j_graph():
    print("\n" + "=" * 60)
    print("BUILDING NEO4J KNOWLEDGE GRAPH")
    print("=" * 60)

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=NEO4J_AUTH,
    )

    try:
        with driver.session() as session:

            # Clear existing graph so the script is idempotent.
            session.run("MATCH (n) DETACH DELETE n")

            for source, rel_type, target in RELATIONSHIPS:

                # Relationship type comes from our trusted
                # hard-coded relationship list.
                query = f"""
                MERGE (a:Company {{ticker: $source}})
                MERGE (b:Company {{ticker: $target}})
                MERGE (a)-[r:{rel_type}]->(b)
                """

                session.run(
                    query,
                    source=source,
                    target=target,
                )

        print(f"Relationships inserted: {len(RELATIONSHIPS)}")

    finally:
        driver.close()

    print("Neo4j graph population complete.")


# ============================================================
# COMPUTE GRAPH FEATURES
# ============================================================

def compute_and_store_graph_features():

    print("\n" + "=" * 60)
    print("COMPUTING GRAPH FEATURES")
    print("=" * 60)

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=NEO4J_AUTH,
    )

    G = nx.DiGraph()

    try:
        with driver.session() as session:

            result = session.run(
                """
                MATCH (a)-[r]->(b)
                RETURN
                    a.ticker AS source,
                    b.ticker AS target
                """
            )

            for record in result:
                G.add_edge(
                    record["source"],
                    record["target"],
                )

    finally:
        driver.close()

    print(f"Graph nodes: {G.number_of_nodes()}")
    print(f"Graph edges: {G.number_of_edges()}")

    if G.number_of_nodes() == 0:
        raise RuntimeError("Neo4j graph is empty.")

    # --------------------------------------------------------
    # Degree Centrality
    # --------------------------------------------------------

    degree_cent = nx.degree_centrality(G)

    # --------------------------------------------------------
    # PageRank
    # --------------------------------------------------------

    pagerank = nx.pagerank(
        G,
        alpha=0.85,
    )

    print("\nGRAPH METRICS")
    print("-" * 60)

    for ticker in sorted(G.nodes()):

        print(
            f"{ticker:6s} | "
            f"degree={degree_cent.get(ticker, 0.0):.6f} | "
            f"pagerank={pagerank.get(ticker, 0.0):.6f}"
        )

    # --------------------------------------------------------
    # Find latest FeatureStore date
    # --------------------------------------------------------

    with pg_engine.connect() as conn:

        latest_date = conn.execute(
            text(
                """
                SELECT MAX(date)
                FROM feature_store
                """
            )
        ).scalar()

    if latest_date is None:
        raise RuntimeError(
            "FeatureStore contains no dates."
        )

    print(
        f"\nUpdating FeatureStore for date: {latest_date}"
    )

    # --------------------------------------------------------
    # Update PostgreSQL
    # --------------------------------------------------------

    updated_rows = 0

    with pg_engine.begin() as conn:

        update_sql = text(
            """
            UPDATE feature_store
            SET
                degree_centrality = :degree_centrality,
                pagerank = :pagerank
            WHERE
                ticker = :ticker
                AND date = :date
            """
        )

        for ticker in G.nodes():

            result = conn.execute(
                update_sql,
                {
                    "ticker": ticker,
                    "date": latest_date,
                    "degree_centrality": degree_cent.get(
                        ticker,
                        0.0,
                    ),
                    "pagerank": pagerank.get(
                        ticker,
                        0.0,
                    ),
                },
            )

            updated_rows += result.rowcount

    print(
        f"FeatureStore rows updated: {updated_rows}"
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    with pg_engine.connect() as conn:

        rows = conn.execute(
            text(
                """
                SELECT
                    ticker,
                    date,
                    degree_centrality,
                    pagerank
                FROM feature_store
                WHERE date = :date
                  AND pagerank > 0
                ORDER BY pagerank DESC
                """
            ),
            {
                "date": latest_date,
            },
        ).fetchall()

    print("\n" + "=" * 60)
    print("POSTGRES GRAPH FEATURES")
    print("=" * 60)

    for row in rows:

        print(
            f"{row.ticker:6s} | "
            f"{row.date} | "
            f"degree={row.degree_centrality:.6f} | "
            f"pagerank={row.pagerank:.6f}"
        )

    print("\nGraph features successfully stored.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_neo4j_graph()

    compute_and_store_graph_features()

    print("\n" + "=" * 60)
    print("KNOWLEDGE GRAPH PIPELINE COMPLETED")
    print("=" * 60)