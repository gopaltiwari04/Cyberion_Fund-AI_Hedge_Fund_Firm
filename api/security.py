"""
Request authentication between the Next.js server and this API.

Browsers never call the API directly. The Next.js route handler verifies the
user's NextAuth (Google) session and forwards the request with:

    X-User-Email, X-Auth-Timestamp, X-Auth-Signature
    signature = hex(HMAC-SHA256(INTERNAL_API_SECRET,
                     "{timestamp}\n{METHOD}\n{path?query}\n{email}\n{sha256(body)}"))

The API recomputes the signature with the shared secret (constant-time
comparison) and rejects requests older than five minutes, so identities can
neither be forged nor replayed later.

Without INTERNAL_API_SECRET:
    * development: requests are attributed to a local developer identity
    * production: only public, read-only endpoints are served; user-specific
      and cost-bearing endpoints return 503 until the secret is configured
"""

from __future__ import annotations

import hashlib
import hmac
import re
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from common.config import get_settings
from common.logs import get_logger

logger = get_logger(__name__)

MAX_SKEW_SECONDS = 300
EMAIL = re.compile(r"^[^@\s]{1,128}@[^@\s]{1,253}\.[^@\s]{2,63}$")
DEV_USER = "developer@localhost.dev"


@dataclass(frozen=True)
class User:
    email: str
    name: str | None = None
    authenticated: bool = True


def canonical_message(timestamp: str, method: str, path_with_query: str, email: str, body: bytes) -> bytes:
    body_hash = hashlib.sha256(body or b"").hexdigest()
    return f"{timestamp}\n{method.upper()}\n{path_with_query}\n{email}\n{body_hash}".encode()


def sign(secret: str, timestamp: str, method: str, path_with_query: str, email: str, body: bytes) -> str:
    return hmac.new(secret.encode(), canonical_message(timestamp, method, path_with_query, email, body), hashlib.sha256).hexdigest()


def _path_with_query(request: Request) -> str:
    query = request.url.query
    return request.url.path + (f"?{query}" if query else "")


async def verify_request(request: Request) -> User | None:
    """Return the verified user, None when unsigned (dev/public), or raise 401."""
    secret = get_settings().internal_api_secret
    email = request.headers.get("x-user-email", "").strip().lower()
    timestamp = request.headers.get("x-auth-timestamp", "")
    signature = request.headers.get("x-auth-signature", "")

    if not secret:
        return None
    if not (email and timestamp and signature):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Authentication required")
    if not EMAIL.match(email):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid identity")
    try:
        skew = abs(time.time() - int(timestamp))
    except ValueError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid timestamp") from None
    if skew > MAX_SKEW_SECONDS:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Request expired")
    body = await request.body()
    expected = sign(secret, timestamp, request.method, _path_with_query(request), email, body)
    if not hmac.compare_digest(expected, signature.lower()):
        logger.warning("Rejected request with invalid signature", extra={"path": request.url.path})
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid signature")
    name = request.headers.get("x-user-name") or None
    return User(email=email, name=name[:120] if name else None)


async def optional_user(request: Request) -> User | None:
    """For read-only endpoints: verified user when signed; open in dev and legacy production mode."""
    user = await verify_request(request)
    request.state.user = user
    return user


async def require_user(request: Request) -> User:
    """For user-specific or cost-bearing endpoints."""
    user = await verify_request(request)
    if user is not None:
        request.state.user = user
        return user
    if get_settings().is_production:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "User features are disabled until INTERNAL_API_SECRET is configured on the API and frontend.",
        )
    request.state.user = User(email=DEV_USER, name="Local developer")
    return request.state.user
