import type { NextRequest } from "next/server";
import { auth } from "@/auth";
import { isAllowedPath, signRequest } from "@/app/lib/server/signing";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_BODY_BYTES = 64 * 1024;
const TIMEOUT_MS = 90_000;

function json(status: number, detail: string) {
  return Response.json({ detail }, { status, headers: { "Cache-Control": "no-store" } });
}

async function proxy(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const session = await auth();
  const email = session?.user?.email?.trim().toLowerCase();
  if (!email) return json(401, "Sign in required");

  const { path } = await context.params;
  if (!isAllowedPath(path)) return json(404, "Not found");

  const backend = process.env.FASTAPI_URL?.replace(/\/$/, "");
  if (!backend) return json(503, "Backend URL is not configured");

  const method = request.method.toUpperCase();
  const body = method === "GET" || method === "HEAD" ? new Uint8Array() : new Uint8Array(await request.arrayBuffer());
  if (body.byteLength > MAX_BODY_BYTES) return json(413, "Request too large");

  const pathWithQuery = `/${path.join("/")}${request.nextUrl.search}`;
  const headers: Record<string, string> = { Accept: "application/json", "X-User-Email": email };
  const name = session?.user?.name?.replace(/[^\x20-\x7E]/g, "").slice(0, 120);
  if (name) headers["X-User-Name"] = name;
  if (body.byteLength) headers["Content-Type"] = "application/json";

  const secret = process.env.INTERNAL_API_SECRET;
  if (secret) {
    const timestamp = Math.floor(Date.now() / 1000).toString();
    headers["X-Auth-Timestamp"] = timestamp;
    headers["X-Auth-Signature"] = signRequest(secret, timestamp, method, pathWithQuery, email, body);
  }

  try {
    const upstream = await fetch(`${backend}${pathWithQuery}`, {
      method,
      headers,
      body: body.byteLength ? body : undefined,
      cache: "no-store",
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "Content-Type": upstream.headers.get("content-type") ?? "application/json",
        "Cache-Control": "no-store",
        ...(upstream.headers.get("retry-after") ? { "Retry-After": upstream.headers.get("retry-after") as string } : {}),
      },
    });
  } catch (error) {
    const timedOut = error instanceof Error && error.name === "TimeoutError";
    return json(timedOut ? 504 : 502, timedOut ? "The analytics service timed out" : "The analytics service is unavailable");
  }
}

export { proxy as GET, proxy as POST, proxy as PUT };
