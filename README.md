# ICP Radar

**Semantic sales intelligence over 3,756 European startups — runs entirely in your browser, costs $0 to host.**

🔗 **Live:** [icp-radar.vercel.app](https://icp-radar.vercel.app/) · 🧠 **How it works (plain language):** [HOW_IT_WORKS.md](./HOW_IT_WORKS.md)

---

## Why this exists

It started with a LinkedIn post about matching candidates to AI startups. The
author's site listed a few thousand European startups; a look at the network tab
showed the underlying data. The interesting question wasn't "can I get the list" —
it was **"what of genuine value can I build from it?"**

The list on its own is someone else's compilation. The value is in what you do
with it. So I turned it into the thing a B2B sales team or a market analyst
actually needs: a tool that takes a *product value proposition* and finds the
companies most likely to be a fit — then helps you act on them.

That's the whole product in one line: **raw data → a ranked, actionable target
market.** Which is the job I'm interested in — turning information into an outcome.

## What it does

- **Semantic ICP search.** Describe what you sell in plain English. It scores all
  3,756 companies by *meaning*, not keywords, and ranks the best 50 with a
  readable fit score.
- **Segmentation.** Filter the matches by funding stage, country, hiring status,
  and founding year.
- **An exec dashboard.** Live stats and breakdowns (top countries, funding-stage
  profile, dominant sector) over your current matches.
- **A Challenger outreach drafter.** For any target, generate a cold email
  structured on the Challenger Sale method (Teach → Tailor → Take Control).
- **CRM export.** One click to a Salesforce/HubSpot-ready CSV.

## The interesting engineering decisions

**The search has no backend.** The query is embedded *in your browser* with a
small open model (`all-MiniLM-L6-v2`, ~23 MB, loaded on first search and cached),
and the similarity maths runs over a precomputed 5.5 MB vector file — client-side,
in a few milliseconds. No search server, no API key, nothing to pay for or attack.

**That wasn't the first design — it's the better one.** The app originally proxied
to a hosted inference API. That endpoint got shut down. Instead of swapping in a
key'd, quota-limited replacement, I moved the model into the browser. The result
is simpler *and* a stronger story: the entire search is genuinely serverless and
$0, and the query is embedded by the same model that built the corpus, so the
scores are directly comparable.

**The one backend call is guarded like production.** Outreach drafting uses Gemini
through a small server route with a per-IP + daily rate limit and a "bring your own
key" bypass — so the public demo works for a visitor without running up an
unbounded bill.

**It's honest about its data.** Website links are name-based searches, not the
source's *guessed* URLs (which could point anywhere). Match scores are labelled as
a calibrated readability scale, not a fake probability. Full reasoning in
[HOW_IT_WORKS.md](./HOW_IT_WORKS.md).

## Stack

Next.js 16 (App Router, TypeScript) · React 19 · transformers.js (in-browser
embeddings) · vanilla CSS · Google Gemini (outreach only) · deploys to Vercel at $0.

---

## Run it locally

```bash
npm install
npm run dev          # http://localhost:3000
```

The search works out of the box (the model downloads from the public HF CDN on
first use). Outreach needs a Gemini key — paste your own in the sidebar, or set one
for the whole app:

```bash
# .env.local  (see .env.example)
GEMINI_API_KEY=your_google_ai_studio_key   # optional; enables the shared-key demo
```

**Regenerating the dataset** (optional — the processed data ships in `public/data/`):
the Python pipeline lives one level up (`../generate_embeddings.py`) and writes
`startups_processed.json` + `embeddings.bin`.

## Deploy

Push to GitHub, import to Vercel, optionally set `GEMINI_API_KEY`, deploy. No other
configuration — there's no database and no build-time secret.
