import { NextResponse } from "next/server";
import { GEMINI_MODEL } from "@/lib/config";

/**
 * Challenger-outreach drafter (Google Gemini).
 *
 * Key strategy (hybrid):
 *  - If the caller supplies their own key (BYOK), we use it and skip all limits.
 *  - Otherwise we fall back to the server key, but guard it like a prod endpoint:
 *    a per-IP rolling limit + a hard daily ceiling so a public demo can't run up
 *    an unbounded bill. When a limit is hit we fail *safely* — a 429 that tells
 *    the user to drop in their own key (which the UI lets them do).
 *
 * Caveat (documented in HOW_IT_WORKS.md): the counters live in module memory, so
 * on serverless they're per-instance and reset on cold start — best-effort, not
 * a guarantee. A production version would use a durable store (Vercel KV / Upstash).
 */

const MAX_FIELD_LEN = 2000; // chars; caps abuse + token cost + injection surface
const WINDOW_MS = 60 * 60 * 1000; // 1-hour rolling window per IP
const MAX_PER_IP = 8; // shared-key requests per IP per window
const MAX_PER_DAY = 100; // shared-key hard ceiling per UTC day

const ipHits = new Map<string, number[]>(); // ip -> request timestamps
let dayStamp = utcDay();
let dayCount = 0;

function utcDay(): string {
  return new Date().toISOString().slice(0, 10);
}

type LimitResult = { ok: true } | { ok: false; reason: "ip" | "daily" };

function checkSharedKeyLimit(ip: string): LimitResult {
  const today = utcDay();
  if (today !== dayStamp) {
    dayStamp = today;
    dayCount = 0;
    ipHits.clear();
  }
  if (dayCount >= MAX_PER_DAY) return { ok: false, reason: "daily" };

  const now = Date.now();
  const recent = (ipHits.get(ip) ?? []).filter((t) => now - t < WINDOW_MS);
  if (recent.length >= MAX_PER_IP) return { ok: false, reason: "ip" };

  recent.push(now);
  ipHits.set(ip, recent);
  dayCount += 1;
  return { ok: true };
}

function cleanField(value: unknown): string {
  return typeof value === "string" ? value.trim().slice(0, MAX_FIELD_LEN) : "";
}

function clientIp(req: Request): string {
  const fwd = req.headers.get("x-forwarded-for");
  return fwd?.split(",")[0]?.trim() || "unknown";
}

export async function POST(req: Request) {
  try {
    const body: unknown = await req.json();
    const {
      productDescription,
      startupName,
      startupDescription,
      startupStage,
      startupCountry,
      userApiKey,
    } = (body ?? {}) as Record<string, unknown>;

    const product = cleanField(productDescription);
    const name = cleanField(startupName);
    if (!product || !name) {
      return NextResponse.json(
        { error: "Product value proposition and startup name are required." },
        { status: 400 },
      );
    }

    const byok = cleanField(userApiKey);
    if (!byok) {
      // Falling back to the shared key — enforce the abuse guard.
      const limit = checkSharedKeyLimit(clientIp(req));
      if (!limit.ok) {
        const msg =
          limit.reason === "daily"
            ? "The shared demo key has hit today's free limit. Add your own Gemini key in the sidebar (🔑 Configure) to keep generating — it stays in your browser."
            : "You've reached the hourly limit on the shared demo key. Add your own Gemini key in the sidebar (🔑 Configure) to continue — it stays in your browser.";
        return NextResponse.json({ error: msg }, { status: 429 });
      }
    }

    const apiKey = byok || process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return NextResponse.json(
        {
          error:
            "No Gemini key available. Add your own in the sidebar (🔑 Configure).",
        },
        { status: 401 },
      );
    }

    const prompt = `You are an elite enterprise B2B sales development representative (SDR) trained in the Challenger Sale methodology (Teach, Tailor, Take Control).

Your goal is to draft a highly personalized, compelling, and professional cold outreach email to a startup.

Here is the context:
1. Our Product Value Proposition (What we sell):
"${product}"

2. Target Startup details:
- Name: ${name}
- What they do/Why they are interesting: "${cleanField(startupDescription)}"
- Funding Stage: ${cleanField(startupStage)}
- Country: ${cleanField(startupCountry)}

Structure the email strictly using the Challenger Sales flow:
1. The Warm-Up: Build immediate credibility. Do NOT start with generic fluff like "I hope this email finds you well" or "Congrats on the funding." Go straight to their space.
2. The Reframe: Introduce a larger industry trend or operational pattern that they might be underestimating.
3. Rational Drowning: Provide data/logical arguments about the cost or risk of not addressing this trend.
4. Emotional Impact: Connect this trend to their specific pain point (e.g. cash flow, efficiency, expansion risk).
5. Value Proposition & Call to Action: Tailor our solution as the remedy to this trend. Offer a short, low-friction, 10-minute chat.

Keep the email short, crisp, and slightly uncomfortable—provoking them to think about their operational leakages. Avoid buzzwords. Write in a direct, professional tone.

Provide your output as a clean text email with a Subject Line.`;

    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-goog-api-key": apiKey,
        },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
        }),
      },
    );

    if (!response.ok) {
      const errText = await response.text();
      // Don't leak the key or raw upstream payload to the client.
      console.error("Gemini API error:", response.status, errText);
      return NextResponse.json(
        { error: `Outreach generation failed (Gemini ${response.status}).` },
        { status: response.status },
      );
    }

    const resData = await response.json();
    const generatedText: unknown =
      resData?.candidates?.[0]?.content?.parts?.[0]?.text;

    if (typeof generatedText !== "string" || !generatedText) {
      return NextResponse.json(
        { error: "Gemini returned an empty response. Try again." },
        { status: 502 },
      );
    }

    return NextResponse.json({ outreachDraft: generatedText });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Internal Server Error";
    console.error("Outreach route error:", message);
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
