/**
 * Client-side query embedding via transformers.js.
 *
 * Why client-side: the HF inference endpoint the app used to call was shut down,
 * and its replacement needs a token + has quota limits. Running the model in the
 * browser keeps the whole search serverless, key-free, and $0 — and means the
 * query vector is produced by the same `all-MiniLM-L6-v2` weights that built the
 * corpus, so cosine similarity stays valid.
 *
 * The model (~23 MB, int8-quantized ONNX) is fetched from the HF CDN on first
 * use and cached by the browser (IndexedDB) for repeat visits. We build it
 * lazily — on the first search, not at page load — so the page paints instantly.
 *
 * Note the dynamic `import()` below: transformers.js is browser-only here, so we
 * load it at call time rather than at module top level. That keeps the heavy ONNX
 * runtime out of the server bundle entirely and stops it running during Next's
 * static prerender. The `import type` is erased at compile time and is free.
 */
import type { FeatureExtractionPipeline } from "@huggingface/transformers";
import { EMBEDDING_MODEL } from "./config";

export type ModelProgress = {
  status: string; // "initiate" | "download" | "progress" | "done" | "ready" | ...
  file?: string;
  progress?: number; // 0–100, present during "progress"
  loaded?: number;
  total?: number;
};

// Singleton: the pipeline is expensive to construct, so we build it once and
// reuse it. The promise is cached so concurrent first-searches don't double-load.
let extractorPromise: Promise<FeatureExtractionPipeline> | null = null;

function getExtractor(
  onProgress?: (p: ModelProgress) => void,
): Promise<FeatureExtractionPipeline> {
  if (!extractorPromise) {
    extractorPromise = (async () => {
      const { pipeline, env } = await import("@huggingface/transformers");
      // Always fetch from the HF CDN; never look for a local /models folder.
      env.allowLocalModels = false;
      return pipeline("feature-extraction", EMBEDDING_MODEL, {
        progress_callback: onProgress,
      });
    })();
  }
  return extractorPromise;
}

/** True once the model has been constructed (cheap, synchronous check). */
export function isModelLoaded(): boolean {
  return extractorPromise !== null;
}

/**
 * Embed a query string into a 384-dim, L2-normalized vector.
 * `pooling: "mean"` + `normalize: true` matches how the corpus was generated,
 * so the dot product against a corpus vector is a true cosine similarity.
 */
export async function embedQuery(
  text: string,
  onProgress?: (p: ModelProgress) => void,
): Promise<Float32Array> {
  const extractor = await getExtractor(onProgress);
  const output = await extractor(text, { pooling: "mean", normalize: true });
  return output.data as Float32Array;
}
