"""
Fixed-window rate limiting per identity (user email, else client IP).

Uses Redis when REDIS_URL is configured (shared across instances) and an
in-process store otherwise (sufficient for a single API instance).
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

from fastapi import HTTPException, Request, status

from common.config import get_settings
from common.logs import get_logger

logger = get_logger(__name__)

LIMITS = {
    "default": (240, 60),
    "recommend": (20, 60),
    "analyst": (8, 60),
    "analyst_daily": (150, 86_400),
    "backtest": (6, 600),
    "write": (60, 60),
}


class _MemoryStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, tuple[int, float]] = defaultdict(lambda: (0, 0.0))

    def hit(self, key: str, window: int) -> int:
        now = time.time()
        with self._lock:
            count, reset = self._counts[key]
            if now >= reset:
                count, reset = 0, now + window
            count += 1
            self._counts[key] = (count, reset)
            if len(self._counts) > 50_000:  # bound memory
                for stale in [k for k, (_, r) in self._counts.items() if r < now][:10_000]:
                    self._counts.pop(stale, None)
            return count


class _RedisStore:
    def __init__(self, url: str) -> None:
        import redis

        self.client = redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
        self.client.ping()

    def hit(self, key: str, window: int) -> int:
        pipe = self.client.pipeline()
        pipe.incr(key)
        pipe.expire(key, window, nx=True)
        return int(pipe.execute()[0])


_store = None


def _get_store():
    global _store
    if _store is None:
        url = get_settings().redis_url
        if url:
            try:
                _store = _RedisStore(url)
            except Exception as error:  # noqa: BLE001
                logger.warning("Redis rate-limit store unavailable, using memory: %s", error)
        if _store is None:
            _store = _MemoryStore()
    return _store


def identity(request: Request) -> str:
    user = getattr(request.state, "user", None)
    if user is not None:
        return f"user:{user.email}"
    forwarded = request.headers.get("x-forwarded-for", "")
    return f"ip:{forwarded.split(',')[0].strip() or (request.client.host if request.client else 'unknown')}"


def enforce(request: Request, bucket: str) -> None:
    limit, window = LIMITS[bucket]
    key = f"rl:{bucket}:{identity(request)}:{int(time.time() // window)}"
    try:
        count = _get_store().hit(key, window)
    except Exception as error:  # noqa: BLE001 - never fail closed on limiter outages
        logger.warning("Rate limiter error: %s", error)
        return
    if count > limit:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Rate limit exceeded for {bucket}; try again later.",
            headers={"Retry-After": str(window)},
        )


def limiter(*buckets: str):
    async def dependency(request: Request) -> None:
        for bucket in buckets:
            enforce(request, bucket)

    return dependency
