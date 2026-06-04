# How ICP Radar Works — plain-language design notes

> Written so I can explain every choice in an interview without hand-waving. For
> each decision: **what** it does, **why** I chose it, and the **trade-off**.
> Audience: technical-but-not-a-specialist (i.e. me). Updated as the build evolves.

---

## 1. What it is, in one breath

You type a product value proposition ("we sell automated invoice auditing for
logistics SMEs"). ICP Radar scores all **3,756 European startups** by how well
they match — semantically, not by keyword — ranks the top 50, lets you filter and
export them, and drafts a Challenger-style cold email for any one of them. It runs
at **$0**, with **no database** and (for the search) **no backend at all**.

---

## 2. The data flow

```
Your text ──► [browser embeds it with a small AI model] ──► a 384-number vector
                                                                   │
   3,756 precomputed vectors (embeddings.bin, 5.5 MB) ◄────────────┘
                                                                   │
                                          dot product × 3,756  (a few ms)
                                                                   │
                              ranked matches ──► filters ──► dashboard / CSV
                                                                   │
                              one match ──► /api/outreach ──► Gemini ──► email
```

Everything except the outreach email happens **in your browser**. The only server
call in the whole app is the optional Gemini draft.

---

## 3. Key decisions

### 3.1 Embeddings (what makes the search "semantic")

**What:** An *embedding* turns a piece of text into a list of numbers (a "vector")
that captures its meaning. Texts about similar things land near each other in this
number-space, even if they share no words. "Carbon accounting for factories" and
"emissions tracking for manufacturers" end up close; "carbon" and "carbohydrate"
do not.

**Why this model (`all-MiniLM-L6-v2`):** It's small (~23 MB), fast, and a proven
default for semantic search. It outputs a **384-dimension** vector per text. Big
enough to be useful, small enough to run in a browser tab.

**Trade-off:** A bigger model would be marginally more accurate but far too heavy
to ship client-side. For ICP matching, MiniLM is more than enough — verified by
the sanity check (below).

### 3.2 Client-side vector search (the architecture headline)

**What:** The query is embedded **in the browser** (transformers.js running the
model via WebAssembly), and the similarity maths runs in the browser too. No
search backend exists.

**Why:** The app originally proxied to a Hugging Face inference API. That endpoint
was **shut down** (the host no longer even resolves), and its replacement requires
an API key and has usage quotas. Rather than re-introduce a key + a paid-ish
dependency, I moved the whole thing into the browser. Now the search is genuinely
serverless, key-free, and free forever — and the query is embedded by the *same
model family* that built the corpus, so the numbers are directly comparable.

**Trade-off:** The model is a **one-time download** on the first search. Specifically, it fetches the `all-MiniLM-L6-v2` model weights (~23 MB) from the public Hugging Face CDN, plus the ONNX WASM runtime (the variant your browser picks can be up to ~23 MB), making the realistic first-search download **≈ 35–47 MB** total (which is then cached in IndexedDB). I lazy-load it on first search — not on page load — so the page still paints instantly, and show a progress bar while it downloads. After the first time, it's instant.

> Engineering detail worth mentioning: transformers.js is imported **dynamically**
> (`await import(...)`), not at the top of the file. That keeps the heavy ML
> runtime out of the *server* bundle entirely and stops it from loading during
> Next.js's static prerender — the library only ever runs in the browser.

### 3.3 Cosine similarity = a dot product (and an honest score)

**What:** To compare two vectors we use **cosine similarity** — essentially "do
these two arrows point the same way?" Because every vector is *normalized* to
length 1 (both the stored ones and the query), cosine similarity is just the **dot
product**: multiply the 384 pairs and add them up. That's why scoring 3,756
companies takes a few milliseconds.

**The honest bit (score calibration):** Raw cosine for this model sits in a
compressed band — even a strong topical match rarely exceeds ~0.55, and unrelated
text still scores ~0.17. Showing the raw number as a percentage would make a great
match look broken ("best fit: 38%"). So the **displayed 0–100% is a linear
rescaling** of cosine (calibrated from the real distribution on this corpus:
noise ≈0.17, top-50 cutoff ≈0.42, best matches ≈0.57–0.65), purely for
readability. **Ranking is always by raw cosine.** It's a presentation transform,
not a probability — and I label it that way rather than dressing it up.

### 3.4 Precomputed corpus (`embeddings.bin`)

**What:** The 3,756 company vectors are computed **once, offline** (Python,
`sentence-transformers`) and saved as a flat binary file of 32-bit floats —
3,756 × 384 × 4 bytes ≈ **5.5 MB**. The browser fetches it once and reads it as a
`Float32Array`.

**Why a binary, not JSON / a database:** A binary float array is ~3× smaller than
the equivalent JSON and loads with zero parsing. No database is needed because the
data is static and small — a vector DB here would be cost and complexity for no
benefit.

**Trade-off:** The corpus is frozen at build time. Updating it means re-running
the Python pipeline and redeploying. For a portfolio/demo that's a feature (zero
infra), not a bug.

### 3.5 Centralized, swappable model ids

**What:** Both model ids (Gemini for outreach, MiniLM for embeddings) live in one
`config.ts`, overridable by env var.

**Why:** This project got burned twice by deprecation — Google retired
`gemini-1.5-flash` (now 404s) and Hugging Face killed the old inference endpoint.
**Model ids are moving targets.** Centralizing them makes the next inevitable swap
a one-line change instead of a hunt.

### 3.6 Outreach: a guarded server route (the only backend)

**What:** Generating a Challenger email calls Google Gemini, via a small server
route (`/api/outreach`). It uses `gemini-2.5-flash-lite` — the cheapest, fastest
current Gemini tier, right for short drafts on a public demo.

**Key strategy (hybrid):**
- If you paste **your own** Gemini key (stored only in your browser's
  localStorage), it's used directly and bypasses all limits.
- Otherwise the demo falls back to a **shared server key**, guarded like a
  production endpoint: a per-IP hourly limit **and** a hard daily ceiling. When a
  limit is hit, it fails *safely* — a clear message telling you to drop in your own
  key (which the UI supports), not a silent error or an unbounded bill.

**Trade-off / honest caveat:** The rate-limit counters live in **server memory**, so on a serverless host they're per-instance and reset on cold start — **best-effort**, not a hard guarantee. A production version would use a durable store (Vercel KV / Upstash). I made that call deliberately: for a capped demo it's enough, and the daily ceiling bounds the worst case. Also: the API key is sent via a request **header**, never a URL query string (URLs get logged), and upstream errors are logged server-side rather than leaked to the client.

**Cost Math & Budget Safety:**
- **Rate Limits:** We set a global cap of `MAX_PER_DAY = 100` calls (globally per instance) and `MAX_PER_IP = 8` calls/hour.
- **Worst-Case Volume:** 100 calls/day × 7 days = 700 calls/week.
- **Token Count & Costs:** Each call uses a prompt template (~450 tokens) + maxed fields (~1,000 tokens) ≈ 1,500 input tokens. A full Challenger cold email draft ≈ 1,000 output tokens.
- **Estimated Cost:** At `gemini-2.5-flash-lite` rates (~$0.10/1M input, ~$0.40/1M output):
  - Cost per call: `(1,500 * $0.10 + 1,000 * $0.40) / 1,000,000 = $0.00055`
  - Cost per week (worst case, single serverless instance): `700 * $0.00055 = $0.385` (≈ €0.36/week).
  - Even with a multi-instance Vercel serverless environment (e.g., an unlikely 10-instance burst), the cost is under ~$4/week.
- **True $0 Guarantee:** The server uses a free-tier Google AI Studio API key. When free limits are exhausted, it rate-limits and fails cleanly, prompting the user to BYOK (Bring Your Own Key), which guarantees exactly $0 in real costs.

### 3.7 Data provenance + the guessed-URL call

**What / why:** The dataset was aggregated from public sources; the **analysis and
tooling are mine**. I publish derived insight (rankings, clusters, aggregates), and
I never frame the raw list as "my dataset."

**The guessed-URL decision:** The source data has no verified website — the
original pipeline *guessed* a URL from the company name + country TLD. Linking
those would risk sending a recruiter to a wrong, parked, or unsafe domain. So
**"Visit site" is a name-based web search**, not a direct link. Zero false
authority, always resolves, honest about what we do and don't know.

---

## 4. What it deliberately does *not* do

- **No database, no auth, no user accounts.** Static data + a stateless route.
- **No raw dataset download.** The app is a search tool, not a data dump.
- **No server-side embedding.** The search has no backend to attack or pay for.

---

## 5. Honest limitations (the caveats I'd raise before someone else does)

- **Quantization:** the browser runs an int8-quantized model; the corpus is fp32.
  Ranking is robust to this — verified the top results for several queries are
  sensible (e.g. "carbon accounting for manufacturers" → Tanso, CarbonChain,
  Normative, all genuine carbon-accounting firms).
- **Short descriptions:** company blurbs are truncated (~160 chars), so clusters
  and matches are topical-but-coarse, not deep.
- **Funding *amounts* are unreliable** (precision artifacts in the source), so any
  analysis leans on funding *stage*, never the amounts.
- **Rate limiting is best-effort** (in-memory; see 3.6).
- **Geographic skew:** the source over-/under-represents some countries; before
  publishing any "hub" claim I validate it isn't just a scraping artifact.
