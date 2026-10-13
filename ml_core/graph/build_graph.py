"""
Financial knowledge graph (Neo4j) and point-in-time graph features.

The graph is built only from sourced data (the earlier hand-written demo
relationships were removed):
    (:Company)-[:IN_SECTOR]->(:Sector)           universe reference data
    (:Company)-[:IN_INDUSTRY]->(:Industry)       SEC SIC code from EDGAR
    (:Company)-[:CO_MENTIONED {date, count}]-(:Company)
                                                 same news article tags both
    (:Company)-[:CORRELATED_WITH {rho, as_of}]-(:Company)
                                                 trailing 60d return correlation

Point-in-time features (stored in feature_store):
    corr_degree_60d       share of other assets with 60d correlation > 0.5
    corr_eigenvector_60d  eigenvector centrality of the |correlation| network
    peer_return_5d        mean 5d return of the 3 most-correlated peers
    kg_degree / kg_pagerank
                          centrality in the knowledge graph using only
                          co-mention edges observed in the trailing 90 days

Run: python -m ml_core.graph.build_graph   (syncs Neo4j)
"""

from __future__ import annotations

from datetime import date

import networkx as nx
import numpy as np
import pandas as pd
from sqlalchemy import text

from common.config import get_settings
from common.db import get_engine
from common.logs import get_logger
from common.market_time import effective_trading_dates
from common.runs import track_run

logger = get_logger(__name__)

CORR_WINDOW = 60
CORR_THRESHOLD = 0.5
PEERS = 3
CO_MENTION_WINDOW_DAYS = 90


def _read(sql: str, **params) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


# ----------------------------------------------------------------------
# Correlation network features
# ----------------------------------------------------------------------

def correlation_features(returns: pd.DataFrame, return_5d: pd.DataFrame) -> pd.DataFrame:
    """returns / return_5d: date x ticker matrices (point-in-time values)."""
    tickers = list(returns.columns)
    values = returns.to_numpy()
    rows = []
    for position in range(len(returns)):
        current_date = returns.index[position]
        if position + 1 < CORR_WINDOW:
            continue
        window = values[position + 1 - CORR_WINDOW: position + 1]
        valid = ~np.isnan(window).any(axis=0)
        if valid.sum() < 3:
            continue
        corr = np.corrcoef(window[:, valid], rowvar=False)
        corr = np.nan_to_num(corr)
        names = [ticker for ticker, ok in zip(tickers, valid) if ok]
        n = len(names)
        absolute = np.abs(corr)
        np.fill_diagonal(absolute, 0.0)
        eigenvalues, eigenvectors = np.linalg.eigh(absolute)
        principal = np.abs(eigenvectors[:, np.argmax(eigenvalues)])
        principal = principal / principal.max() if principal.max() > 0 else principal
        degree = ((corr > CORR_THRESHOLD).sum(axis=1) - 1) / max(n - 1, 1)
        r5 = return_5d.iloc[position][names].to_numpy()
        for i, ticker in enumerate(names):
            order = [j for j in np.argsort(-corr[i]) if j != i][:PEERS]
            peer_values = r5[order]
            rows.append(
                {
                    "date": current_date,
                    "ticker": ticker,
                    "corr_degree_60d": float(degree[i]),
                    "corr_eigenvector_60d": float(principal[i]),
                    "peer_return_5d": float(np.nanmean(peer_values)) if np.isfinite(peer_values).any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Knowledge graph edges
# ----------------------------------------------------------------------

def static_edges(assets: pd.DataFrame) -> list[tuple[str, str, str]]:
    edges = []
    for row in assets.itertuples():
        if row.sector:
            edges.append((row.ticker, f"sector:{row.sector}", "IN_SECTOR"))
        if row.sic_code:
            edges.append((row.ticker, f"sic:{row.sic_code}", "IN_INDUSTRY"))
    return edges


def co_mention_events(news: pd.DataFrame, universe: set[str], trading_dates) -> pd.DataFrame:
    """One row per (ticker_a, ticker_b, effective_date) where an article tags both companies."""
    if news.empty:
        return pd.DataFrame(columns=["a", "b", "date", "count"])
    news = news[news["ticker"].isin(universe)].copy()
    # NewsAPI legacy rows carry one ticker per row; identical headlines across tickers count as one article.
    news["article"] = np.where(news["uuid"].str.startswith("newsapi:"), news["title"].fillna(""), news["uuid"])
    news["date"] = effective_trading_dates(news["published_at"], trading_dates)
    news = news.dropna(subset=["date"])
    pairs = []
    for (_, event_date), group in news.groupby(["article", "date"]):
        names = sorted(set(group["ticker"]))
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append((names[i], names[j], event_date))
    if not pairs:
        return pd.DataFrame(columns=["a", "b", "date", "count"])
    frame = pd.DataFrame(pairs, columns=["a", "b", "date"])
    return frame.groupby(["a", "b", "date"]).size().rename("count").reset_index()


def knowledge_graph_features(dates: list[date], tickers: list[str], edges: list[tuple[str, str, str]],
                             mentions: pd.DataFrame) -> pd.DataFrame:
    def centrality(mention_edges: pd.DataFrame) -> dict[str, tuple[float, float]]:
        graph = nx.Graph()
        graph.add_nodes_from(tickers)
        graph.add_edges_from((a, b) for a, b, _ in edges)
        for row in mention_edges.itertuples():
            weight = graph.get_edge_data(row.a, row.b, {}).get("weight", 0) + row.count
            graph.add_edge(row.a, row.b, weight=weight)
        degree = nx.degree_centrality(graph)
        rank = nx.pagerank(graph, alpha=0.85)
        return {ticker: (degree.get(ticker, 0.0), rank.get(ticker, 0.0)) for ticker in tickers}

    mentions = mentions.copy()
    if not mentions.empty:
        mentions["date"] = pd.to_datetime(mentions["date"])
    static = centrality(mentions.iloc[0:0])
    first_mention = mentions["date"].min() if not mentions.empty else None
    rows = []
    for current in sorted(set(pd.to_datetime(dates))):
        if first_mention is None or current < first_mention:
            values = static
        else:
            window = mentions[(mentions["date"] <= current) & (mentions["date"] > current - pd.Timedelta(days=CO_MENTION_WINDOW_DAYS))]
            values = centrality(window) if not window.empty else static
        rows.extend({"date": current, "ticker": t, "kg_degree": v[0], "kg_pagerank": v[1]} for t, v in values.items())
    return pd.DataFrame(rows)


def graph_features(frame: pd.DataFrame) -> pd.DataFrame:
    """frame needs ticker, date, return_1d, return_5d (from the technical feature step)."""
    data = frame[["ticker", "date", "return_1d", "return_5d"]].copy()
    data["date"] = pd.to_datetime(data["date"])
    returns = data.pivot_table(index="date", columns="ticker", values="return_1d").sort_index()
    return_5d = data.pivot_table(index="date", columns="ticker", values="return_5d").sort_index()
    corr = correlation_features(returns, return_5d.reindex(returns.index))

    tickers = sorted(data["ticker"].unique())
    assets = _read("SELECT ticker, sector, sic_code FROM assets WHERE ticker = ANY(:t)", t=tickers)
    news = _read("SELECT uuid, ticker, title, published_at FROM financial_news WHERE ticker IS NOT NULL")
    trading_dates = returns.index.date
    mentions = co_mention_events(news, set(tickers), trading_dates)
    kg = knowledge_graph_features(list(trading_dates), tickers, static_edges(assets), mentions)

    result = data[["ticker", "date"]].merge(corr, on=["ticker", "date"], how="left")
    return result.merge(kg, on=["ticker", "date"], how="left")


# ----------------------------------------------------------------------
# Neo4j synchronisation
# ----------------------------------------------------------------------

def sync_neo4j() -> dict:
    from neo4j import GraphDatabase

    settings = get_settings()
    assets = _read("SELECT ticker, name, asset_class, sector, sic_code, sic_description FROM assets WHERE is_active")
    news = _read("SELECT uuid, ticker, title, published_at FROM financial_news WHERE ticker IS NOT NULL")
    prices = _read("SELECT ticker, date, close FROM market_data WHERE date >= now() - interval '200 days'")
    tickers = set(assets["ticker"])
    trading_dates = sorted(pd.to_datetime(prices["date"]).dt.date.unique())
    mentions = co_mention_events(news, tickers, trading_dates)

    matrix = prices.pivot_table(index="date", columns="ticker", values="close").sort_index().pct_change().tail(CORR_WINDOW)
    corr = matrix.corr()
    as_of = str(matrix.index.max()) if len(matrix) else None

    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    try:
        with driver.session() as session:
            session.run("CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE")
            # Remove the hand-written demo relationships from the earlier prototype.
            session.run("MATCH (:Company)-[r:SUPPLIES|PARTNERS_WITH|COMPETES_WITH|CONTAINS]->() DELETE r")
            session.run("MATCH (c:Company) WHERE NOT c.ticker IN $tickers DETACH DELETE c", tickers=list(tickers))
            for row in assets.itertuples():
                session.run(
                    """
                    MERGE (c:Company {ticker: $ticker})
                    SET c.name = $name, c.asset_class = $asset_class
                    MERGE (s:Sector {name: $sector})
                    MERGE (c)-[:IN_SECTOR]->(s)
                    """,
                    ticker=row.ticker, name=row.name, asset_class=row.asset_class, sector=row.sector,
                )
                if row.sic_code:
                    session.run(
                        """
                        MATCH (c:Company {ticker: $ticker})
                        MERGE (i:Industry {sic: $sic}) SET i.description = $description
                        MERGE (c)-[:IN_INDUSTRY]->(i)
                        """,
                        ticker=row.ticker, sic=row.sic_code, description=row.sic_description,
                    )
            for row in mentions.itertuples():
                session.run(
                    """
                    MATCH (a:Company {ticker: $a}), (b:Company {ticker: $b})
                    MERGE (a)-[r:CO_MENTIONED {date: date($date)}]->(b)
                    SET r.count = $count
                    """,
                    a=row.a, b=row.b, date=str(row.date), count=int(row.count),
                )
            session.run("MATCH ()-[r:CORRELATED_WITH]->() DELETE r")
            for a in corr.columns:
                for b in corr.columns:
                    if a < b and pd.notna(corr.loc[a, b]) and corr.loc[a, b] > CORR_THRESHOLD:
                        session.run(
                            """
                            MATCH (x:Company {ticker: $a}), (y:Company {ticker: $b})
                            MERGE (x)-[r:CORRELATED_WITH]->(y) SET r.rho = $rho, r.as_of = $as_of
                            """,
                            a=a, b=b, rho=float(corr.loc[a, b]), as_of=as_of,
                        )
            counts = session.run("MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() RETURN nodes, count(r) AS edges").single()
    finally:
        driver.close()
    return {"nodes": counts["nodes"], "edges": counts["edges"], "co_mention_events": len(mentions)}


def main() -> None:
    with track_run("knowledge_graph", "sync_neo4j") as ctx:
        ctx.details.update(sync_neo4j())


if __name__ == "__main__":
    main()
