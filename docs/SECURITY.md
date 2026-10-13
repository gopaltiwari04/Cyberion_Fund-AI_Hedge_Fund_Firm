# Security

## Authentication & authorisation

* Users sign in with Google (NextAuth v5, JWT sessions). The Next.js proxy (`frontend/proxy.ts`) protects
  every page; `/login` is the only public page.
* The browser never calls FastAPI. `frontend/app/api/[...path]/route.ts` verifies the session, allow-lists
  API areas and path characters, limits body size (64 KB), and signs each request:
  `HMAC-SHA256(INTERNAL_API_SECRET, ts \n METHOD \n path?query \n email \n sha256(body))`.
* FastAPI (`api/security.py`) recomputes the signature in constant time and rejects requests older than
  5 minutes, so identities cannot be forged or replayed. User data is always scoped by the verified email.
* Without the secret: development uses a local developer identity; production disables user/cost endpoints.

## Input validation & errors

* All request bodies are strict Pydantic models (`extra="forbid"`, bounded numbers, universe-checked tickers,
  date ranges). SQL uses bound parameters only.
* Errors return generic messages with a request id; database errors return 503 without internals.
  (The previous API returned raw exception strings.)
* Security headers on API and dashboard responses; OpenAPI docs are disabled in production.

## Abuse controls

* Per-identity rate limits (`api/ratelimit.py`): default 240/min; recommendations 20/min; analyst 8/min and
  150/day (LLM cost control); backtests 6 per 10 min and one concurrent run per user; bounded worker pool.
* SEC analyst: passages are treated as data (prompt-injection guidance in the system prompt) and the server
  discards statements whose citations do not map to retrieved sources.

## Secrets

* `.env`, `frontend/.env.local` and dumps are git-ignored; `.env.example` files document every variable
  without values. Docker defaults are for local development only.
* SEC requests carry the declared `SEC_USER_AGENT`; API keys are sent in headers, never in logged URLs
  (except FRED/Marketaux, whose APIs require query parameters; those URLs are not logged).

## Audit trail

`recommendation_runs` (inputs + full output + model id), `research_queries`, `backtest_runs` (config hash,
code version), `pipeline_runs`, and the model registry (data hash, code version, metrics).
