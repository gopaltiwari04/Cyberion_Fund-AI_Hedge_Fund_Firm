import { createHash, createHmac } from "node:crypto";

/**
 * Canonical request signature shared with the FastAPI service (api/security.py):
 * HMAC-SHA256(secret, "{timestamp}\n{METHOD}\n{path?query}\n{email}\n{sha256(body)}").
 */
export function signRequest(
  secret: string,
  timestamp: string,
  method: string,
  pathWithQuery: string,
  email: string,
  body: Uint8Array,
): string {
  const bodyHash = createHash("sha256").update(body).digest("hex");
  const message = `${timestamp}\n${method.toUpperCase()}\n${pathWithQuery}\n${email}\n${bodyHash}`;
  return createHmac("sha256", secret).update(message).digest("hex");
}

const ALLOWED_PREFIXES = ["health", "markets", "news", "portfolio", "risk", "me", "research", "analyst", "backtests", "system"];

/** Only known API areas are proxied; path segments are restricted to safe characters. */
export function isAllowedPath(segments: string[]): boolean {
  if (segments.length === 0 || segments.length > 6) return false;
  if (!ALLOWED_PREFIXES.includes(segments[0])) return false;
  return segments.every((segment) => /^[A-Za-z0-9._-]{1,64}$/.test(segment) && segment !== ".." && segment !== ".");
}
