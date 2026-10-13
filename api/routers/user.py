"""Per-user profile, holdings and personalised (non-advisory) recommendations."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.deps import get_db, jsonable
from api.ratelimit import limiter
from api.schemas import HoldingsUpdate, ProfileUpdate, RecommendationRequest
from api.security import User, require_user
from common.logs import get_logger
from ml_core.portfolio.market import latest_prices
from ml_core.portfolio.recommend import build_recommendation

logger = get_logger(__name__)
router = APIRouter(prefix="/me", tags=["user"])

PROFILE_COLUMNS = "email, display_name, risk_profile, cvar_limit, max_weight, max_turnover, transaction_cost_bps, cash_balance, updated_at"


def ensure_profile(db: Session, user: User) -> dict:
    db.execute(
        text(
            "INSERT INTO user_profiles (email, display_name) VALUES (:e, :n) ON CONFLICT (email) DO NOTHING"
        ),
        {"e": user.email, "n": user.name},
    )
    db.commit()
    return dict(db.execute(text(f"SELECT {PROFILE_COLUMNS} FROM user_profiles WHERE email = :e"), {"e": user.email}).mappings().one())


def holdings_for(db: Session, email: str) -> list[dict]:
    rows = db.execute(
        text("SELECT ticker, quantity, average_cost, updated_at FROM user_holdings WHERE email = :e ORDER BY ticker"),
        {"e": email},
    ).mappings().all()
    return [dict(row) for row in rows]


@router.get("/profile")
def get_profile(user: User = Depends(require_user), db: Session = Depends(get_db)):
    return jsonable(ensure_profile(db, user))


@router.put("/profile", dependencies=[Depends(limiter("write"))])
def update_profile(payload: ProfileUpdate, user: User = Depends(require_user), db: Session = Depends(get_db)):
    ensure_profile(db, user)
    db.execute(
        text(
            """
            UPDATE user_profiles SET display_name = COALESCE(:display_name, display_name), risk_profile = :risk_profile,
                cvar_limit = :cvar_limit, max_weight = :max_weight, max_turnover = :max_turnover,
                transaction_cost_bps = :transaction_cost_bps, cash_balance = :cash_balance, updated_at = now()
            WHERE email = :email
            """
        ),
        {**payload.model_dump(), "email": user.email},
    )
    db.commit()
    return jsonable(ensure_profile(db, user))


@router.get("/holdings")
def get_holdings(user: User = Depends(require_user), db: Session = Depends(get_db)):
    profile = ensure_profile(db, user)
    holdings = holdings_for(db, user.email)
    prices = latest_prices([row["ticker"] for row in holdings]) if holdings else {}
    enriched, total = [], float(profile["cash_balance"])
    for row in holdings:
        price = prices.get(row["ticker"])
        value = row["quantity"] * price["close"] if price else None
        total += value or 0.0
        pnl = (price["close"] - row["average_cost"]) * row["quantity"] if price and row["average_cost"] else None
        enriched.append({**row, "price": price["close"] if price else None, "price_date": price["date"] if price else None,
                         "market_value": value, "unrealized_pnl": pnl})
    for row in enriched:
        row["weight"] = (row["market_value"] / total) if total and row["market_value"] is not None else None
    return jsonable({"cash_balance": profile["cash_balance"], "total_value": total, "holdings": enriched})


@router.put("/holdings", dependencies=[Depends(limiter("write"))])
def replace_holdings(payload: HoldingsUpdate, user: User = Depends(require_user), db: Session = Depends(get_db)):
    ensure_profile(db, user)
    db.execute(text("DELETE FROM user_holdings WHERE email = :e"), {"e": user.email})
    rows = [{"e": user.email, **holding.model_dump()} for holding in payload.holdings if holding.quantity > 0]
    if rows:
        db.execute(
            text(
                "INSERT INTO user_holdings (email, ticker, quantity, average_cost, updated_at) "
                "VALUES (:e, :ticker, :quantity, :average_cost, now())"
            ),
            rows,
        )
    if payload.cash_balance is not None:
        db.execute(text("UPDATE user_profiles SET cash_balance = :c, updated_at = now() WHERE email = :e"),
                   {"c": payload.cash_balance, "e": user.email})
    db.commit()
    return get_holdings(user, db)


@router.post("/recommendations", dependencies=[Depends(limiter("recommend"))])
def create_recommendation(payload: RecommendationRequest, user: User = Depends(require_user), db: Session = Depends(get_db)):
    profile = ensure_profile(db, user)
    holdings = {row["ticker"]: row["quantity"] for row in holdings_for(db, user.email)}
    overrides = {
        "cvar_limit": payload.cvar_limit if payload.cvar_limit is not None else profile["cvar_limit"],
        "max_weight": payload.max_weight if payload.max_weight is not None else profile["max_weight"],
        "max_turnover": payload.max_turnover if payload.max_turnover is not None else profile["max_turnover"],
        "transaction_cost_bps": (payload.transaction_cost_bps if payload.transaction_cost_bps is not None
                                 else profile["transaction_cost_bps"]),
    }
    if not holdings and float(profile["cash_balance"]) <= 0:
        raise HTTPException(status_code=400, detail="Add holdings or a cash balance before requesting recommendations.")
    try:
        result = build_recommendation(
            holdings, float(profile["cash_balance"]), payload.risk_profile or profile["risk_profile"],
            payload.horizon_days, overrides,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
    result = jsonable(result)
    run_id = uuid.uuid4()
    db.execute(
        text(
            "INSERT INTO recommendation_runs (id, email, model_id, request, result) "
            "VALUES (:id, :e, :m, CAST(:req AS jsonb), CAST(:res AS jsonb))"
        ),
        {"id": run_id, "e": user.email, "m": (result.get("model") or {}).get("id"),
         "req": json.dumps({**payload.model_dump(), "holdings": holdings, "cash": profile["cash_balance"]}),
         "res": json.dumps(result)},
    )
    db.commit()
    return {"id": str(run_id), **result}


@router.get("/recommendations")
def list_recommendations(user: User = Depends(require_user), db: Session = Depends(get_db)):
    rows = db.execute(
        text(
            """
            SELECT id, created_at, result->>'status' AS status, result->>'as_of_date' AS as_of_date,
                   result->'proposal'->'metrics'->>'expected_return_horizon' AS expected_return,
                   result->'proposal'->'metrics'->>'cvar_95_1d' AS cvar
            FROM recommendation_runs WHERE email = :e ORDER BY created_at DESC LIMIT 20
            """
        ),
        {"e": user.email},
    ).mappings().all()
    return {"runs": [jsonable(dict(row)) for row in rows]}


@router.get("/recommendations/{run_id}")
def get_recommendation(run_id: uuid.UUID, user: User = Depends(require_user), db: Session = Depends(get_db)):
    row = db.execute(
        text("SELECT id, created_at, result FROM recommendation_runs WHERE id = :id AND email = :e"),
        {"id": run_id, "e": user.email},
    ).mappings().one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {"id": str(row["id"]), "created_at": row["created_at"].isoformat(), **row["result"]}
