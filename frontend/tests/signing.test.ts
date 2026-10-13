import { describe, expect, it } from "vitest";
import { isAllowedPath, signRequest } from "../app/lib/server/signing";

describe("signRequest", () => {
  // Reference values produced by the Python verifier (api.security.sign) for the same inputs.
  it("matches the FastAPI signature for a POST with body and query", () => {
    const body = new TextEncoder().encode('{"horizon_days":5}');
    expect(signRequest("vector-secret", "1760000000", "post", "/me/recommendations?x=1", "user@example.com", body))
      .toBe("52c91501ca78fab72b4e86888ae87777e03d7f5642de37e8f4d8ddf9dda390fd");
  });

  it("matches the FastAPI signature for a GET without body", () => {
    expect(signRequest("vector-secret", "1760000000", "GET", "/markets", "user@example.com", new Uint8Array()))
      .toBe("809daf5c66a5baa906a5eb63880a3dd830b0b06f40f5ca5a03488f1a4061f77d");
  });

  it("changes when any signed component changes", () => {
    const base = signRequest("s", "1", "GET", "/markets", "a@b.co", new Uint8Array());
    expect(signRequest("s", "1", "GET", "/markets", "x@b.co", new Uint8Array())).not.toBe(base);
    expect(signRequest("s", "2", "GET", "/markets", "a@b.co", new Uint8Array())).not.toBe(base);
    expect(signRequest("t", "1", "GET", "/markets", "a@b.co", new Uint8Array())).not.toBe(base);
  });
});

describe("isAllowedPath", () => {
  it("allows known API areas", () => {
    expect(isAllowedPath(["markets"])).toBe(true);
    expect(isAllowedPath(["markets", "AAPL", "forecast"])).toBe(true);
    expect(isAllowedPath(["backtests", "0fe7269e-e34e-40dd-8b5b-c89ee3221fe4"])).toBe(true);
  });

  it("rejects unknown areas, traversal and odd characters", () => {
    expect(isAllowedPath([])).toBe(false);
    expect(isAllowedPath(["docs"])).toBe(false);
    expect(isAllowedPath(["openapi.json"])).toBe(false);
    expect(isAllowedPath(["markets", ".."])).toBe(false);
    expect(isAllowedPath(["markets", "a b"])).toBe(false);
    expect(isAllowedPath(["markets", "%2e%2e"])).toBe(false);
  });
});
