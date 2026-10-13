import { NextRequest } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { signRequest } from "../app/lib/server/signing";

const session = { current: null as null | { user: { email: string; name: string } } };
vi.mock("@/auth", () => ({ auth: async () => session.current }));

const { GET, POST } = await import("../app/api/[...path]/route");

function context(path: string[]) {
  return { params: Promise.resolve({ path }) };
}

describe("API proxy route", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    process.env.FASTAPI_URL = "http://backend.test";
    process.env.INTERNAL_API_SECRET = "proxy-secret";
    fetchMock.mockReset();
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ ok: true }), { status: 200, headers: { "content-type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    session.current = null;
  });

  it("rejects unauthenticated requests without contacting the backend", async () => {
    const response = await GET(new NextRequest("http://localhost/api/markets"), context(["markets"]));
    expect(response.status).toBe(401);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("returns 404 for paths outside the allow-list", async () => {
    session.current = { user: { email: "a@b.co", name: "A" } };
    const response = await GET(new NextRequest("http://localhost/api/openapi.json"), context(["openapi.json"]));
    expect(response.status).toBe(404);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("forwards signed requests with the verified identity", async () => {
    session.current = { user: { email: "Analyst@Example.com", name: "Ana Lyst" } };
    const body = JSON.stringify({ horizon_days: 5 });
    const request = new NextRequest("http://localhost/api/me/recommendations?x=1", { method: "POST", body });
    const response = await POST(request, context(["me", "recommendations"]));
    expect(response.status).toBe(200);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://backend.test/me/recommendations?x=1");
    const headers = init.headers as Record<string, string>;
    expect(headers["X-User-Email"]).toBe("analyst@example.com");
    const expected = signRequest("proxy-secret", headers["X-Auth-Timestamp"], "POST", "/me/recommendations?x=1",
      "analyst@example.com", new TextEncoder().encode(body));
    expect(headers["X-Auth-Signature"]).toBe(expected);
  });

  it("maps backend outages to 502 without leaking details", async () => {
    session.current = { user: { email: "a@b.co", name: "A" } };
    fetchMock.mockRejectedValue(new Error("connect ECONNREFUSED 10.0.0.1:8000"));
    const response = await GET(new NextRequest("http://localhost/api/markets"), context(["markets"]));
    expect(response.status).toBe(502);
    expect(JSON.stringify(await response.json())).not.toContain("ECONNREFUSED");
  });
});
