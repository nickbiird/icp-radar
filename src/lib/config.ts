/**
 * Central config — single source of truth for things that drift.
 *
 * Model ids deprecate without much warning. This project already lost two:
 * Google retired `gemini-1.5-flash` (now 404s) and Hugging Face shut down the
 * legacy `api-inference.huggingface.co` endpoint the app used to call. Both are
 * pinned here so swapping a model is a one-line change, overridable by env.
 */

// Server-side only (used in the /api/outreach route). 2.5-flash-lite is the
// cheapest/fastest current Gemini tier — right for a rate-limited public demo.
export const GEMINI_MODEL = process.env.GEMINI_MODEL ?? "gemini-2.5-flash-lite";

// Client-side: the embedding model that runs in the browser via transformers.js.
// MUST stay the same family as the corpus (`all-MiniLM-L6-v2`, 384-dim) so the
// query vector is comparable to the precomputed vectors in embeddings.bin.
export const EMBEDDING_MODEL =
  process.env.NEXT_PUBLIC_EMBEDDING_MODEL ?? "Xenova/all-MiniLM-L6-v2";

// Embedding dimensionality of all-MiniLM-L6-v2.
export const EMBEDDING_DIM = 384;

// How many top matches we surface (and compute dashboard stats over).
export const TOP_K = 50;

/**
 * Cosine similarity for MiniLM sits in a compressed band — even a strong topical
 * match rarely exceeds ~0.55, and unrelated text still scores ~0.1. Showing the
 * raw cosine as a percentage makes great matches look broken ("best fit: 38%").
 * We linearly rescale [LOW, HIGH] → [0, 100] purely for readability; ranking is
 * always by the raw cosine. Calibrated from the observed distribution on this
 * corpus (noise floor ≈0.17, top-50 cutoff ≈0.42, best matches ≈0.57–0.65), so a
 * strong match reads ~90–100%, a mid-list match ~60%, and noise ~5%. This is a
 * presentation transform, not a probability. See HOW_IT_WORKS.md.
 */
export const SCORE_CALIBRATION = { low: 0.15, high: 0.6 } as const;
