"use client";

import React, { useState, useEffect, useRef, useMemo } from "react";
import { embedQuery, isModelLoaded, type ModelProgress } from "@/lib/embedder";
import { EMBEDDING_DIM, TOP_K, SCORE_CALIBRATION } from "@/lib/config";

interface Startup {
  Name: string;
  Description: string;
  "Stage / Funding Tier": string;
  City: string;
  Country: string;
  "Website URL": string;
  "Is Hiring Now": boolean;
  "Startup ID": string;
  Domain: string;
  "Year Founded": number | string;
  Accelerators: string;
  "Last Funding Amount": number | string;
  // Added by analysis/enrich_dataset.py — the offline k-means assignment brought online.
  clusterId: number;
  clusterName: string;
  score?: number;
  rawScore?: number;
}

// Per-cluster aggregates computed once from the loaded dataset — the "model
// output" the Market Map and the drill-down insights panel render.
interface ClusterStat {
  id: number;
  name: string;
  size: number;
  sharePct: number;
  stageCounts: Record<string, number>;
  topCountries: { name: string; value: number }[];
  topDomains: { name: string; value: number }[];
  distinctDomains: number; // domains holding ≥10% of the cluster
  crossTaxonomy: boolean; // ≥3 source categories merged into one theme
}

// One-line editorial read per theme, distilled from analysis/FINDINGS.md. The
// numbers on screen are computed live; these lines carry the interpretation.
const CLUSTER_TAGLINES: Record<string, string> = {
  "AI & Machine Learning":
    "The largest and earliest theme — most firms are Pre-Seed/Seed. A crowded, capital-hungry land-grab.",
  Fintech:
    "Europe's deepest software vertical — unusually mature, Series A/B heavy, UK-led.",
  "Commerce Operations":
    "Four filed categories, one job: tooling that runs a transactional business. Spain co-leads.",
  "Health & Medtech":
    "Clinical and patient-facing software — broad, seed-stage, geographically spread.",
  "Technical Workflow Layer":
    "Devtools + compliance + no-code merged: software that removes manual workflow friction.",
  "Deep & Physical Tech":
    "Hardware, robotics, aerospace, IoT — the German-Swiss 'atoms, not bits' cluster.",
  "Media, EdTech & Sports":
    "Consumer attention and learning — Spain's #1 theme by company count.",
  "Travel, Mobility & Logistics":
    "Moving people and goods — another Spanish-led, operational cluster.",
  Energy:
    "The grid and its software — Nordic/German, infrastructure-heavy and capital-intensive.",
  Biotech:
    "Patient capital: the only theme where Series A outnumbers Seed. CH/SE/DK concentrated.",
  "Climate Tech":
    "A recent rush — half sit at Seed, still proving the model. The mirror image of biotech.",
  "HR & People Tech":
    "Hiring, payroll and talent software — distributed across NL/DE, mostly mid-stage.",
  Cybersecurity:
    "Security and trust software — Spain co-leads, steady mid-stage funding.",
  "Data & Analytics":
    "The data plumbing layer — small, NL/UK-led; the closest analogue to this tool itself.",
  "Agritech & Foodtech":
    "Food and farming tech — Series-A-heavy, Spain co-leads on count.",
};

// Maturity buckets in canonical order, each with a colour for the stacked bar.
// First matching bucket wins, so "Series C+" must precede the "Other" catch-all.
const MATURITY_BUCKETS: { label: string; color: string; match: (s: string) => boolean }[] = [
  { label: "Pre-Seed", color: "#475569", match: (s) => s === "Pre-Seed" },
  { label: "Seed", color: "#10b981", match: (s) => s === "Seed" },
  { label: "Series A", color: "#3b82f6", match: (s) => s === "Series A" },
  { label: "Series B", color: "#8b5cf6", match: (s) => s === "Series B" },
  { label: "Series C+", color: "#f59e0b", match: (s) => /^Series [C-Z]/.test(s) },
  { label: "Other", color: "#273349", match: () => true },
];

interface MaturitySegment {
  label: string;
  color: string;
  value: number;
  pct: number;
}

// Collapse raw stage counts into the canonical maturity segments (with %).
function bucketStages(stageCounts: Record<string, number>): MaturitySegment[] {
  const total = Object.values(stageCounts).reduce((a, b) => a + b, 0) || 1;
  const values = MATURITY_BUCKETS.map(() => 0);
  for (const [stage, count] of Object.entries(stageCounts)) {
    const idx = MATURITY_BUCKETS.findIndex((b) => b.match(stage));
    values[idx >= 0 ? idx : MATURITY_BUCKETS.length - 1] += count;
  }
  return MATURITY_BUCKETS.map((b, i) => ({
    label: b.label,
    color: b.color,
    value: values[i],
    pct: Math.round((values[i] / total) * 100),
  }));
}

function topEntries(counts: Record<string, number>, k: number) {
  return Object.entries(counts)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, k);
}

function errMessage(e: unknown): string {
  return e instanceof Error ? e.message : "Something went wrong.";
}

// We deliberately do NOT link the algorithmically-guessed "Website URL" in the
// dataset — a wrong/squatted domain would be a credibility risk. A name-based
// search carries no false authority and always resolves. (See HOW_IT_WORKS.md.)
function companySearchUrl(s: Startup): string {
  const q = [s.Name, s.Country, "startup"].filter(Boolean).join(" ");
  return `https://www.google.com/search?q=${encodeURIComponent(q)}`;
}

export default function Home() {
  // Data Refs for performance (static database)
  const embeddingsRef = useRef<Float32Array | null>(null);
  const metadataRef = useRef<Startup[]>([]);
  const mainRef = useRef<HTMLDivElement>(null);

  // Page States
  const [loadingData, setLoadingData] = useState<boolean>(true);
  const [dataReady, setDataReady] = useState<boolean>(false);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [query, setQuery] = useState<string>("");
  const [searchedQuery, setSearchedQuery] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  // Client-side embedding model loading (lazy: built on first search)
  const [modelState, setModelState] = useState<"idle" | "loading" | "ready">(
    "idle",
  );
  const [modelProgress, setModelProgress] = useState<number>(0);

  // Lists & Metadata Stats
  const [totalStartupsCount, setTotalStartupsCount] = useState<number>(0);
  const [countryCount, setCountryCount] = useState<number>(0);
  const [corpus, setCorpus] = useState<Startup[]>([]); // render-time browse base
  const [allScoredResults, setAllScoredResults] = useState<Startup[]>([]);
  const [availableStages, setAvailableStages] = useState<string[]>([]);
  const [availableCountries, setAvailableCountries] = useState<string[]>([]);
  const [clusterStats, setClusterStats] = useState<ClusterStat[]>([]);

  // Filter States
  const [filterStage, setFilterStage] = useState<string>("");
  const [filterCountry, setFilterCountry] = useState<string>("");
  const [filterHiring, setFilterHiring] = useState<string>(""); // "" | "hiring" | "not_hiring"
  const [filterMinYear, setFilterMinYear] = useState<string>("");
  const [filterCluster, setFilterCluster] = useState<string>(""); // clusterName | ""

  // UI Selection & Keys
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [userApiKey, setUserApiKey] = useState<string>("");
  const [hasSavedKey, setHasSavedKey] = useState<boolean>(false);
  const [showKeyInput, setShowKeyInput] = useState<boolean>(false);

  // Modal / Outreach States
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [activeStartup, setActiveStartup] = useState<Startup | null>(null);
  const [isGeneratingDraft, setIsGeneratingDraft] = useState<boolean>(false);
  const [outreachDraft, setOutreachDraft] = useState<string>("");
  const [copyFeedback, setCopyFeedback] = useState<boolean>(false);

  // Guards against stale async results overwriting a newer search
  const searchSeq = useRef<number>(0);
  const modalRef = useRef<HTMLDivElement>(null);

  // Load datasets on mount
  useEffect(() => {
    async function loadDataset() {
      try {
        setLoadingData(true);
        setError(null);

        // Restore a previously-saved Gemini key from localStorage (client-only).
        const savedKey = localStorage.getItem("gemini_api_key");
        if (savedKey) {
          setUserApiKey(savedKey);
          setHasSavedKey(true);
        }

        // Fetch metadata JSON
        const metaRes = await fetch("/data/startups_processed.json");
        if (!metaRes.ok) throw new Error("Metadata JSON not found");
        const metaData = (await metaRes.json()) as Startup[];
        metadataRef.current = metaData;
        setCorpus(metaData);
        setTotalStartupsCount(metaData.length);

        // Fetch binary embeddings float32 matrix
        const binRes = await fetch("/data/embeddings.bin");
        if (!binRes.ok) throw new Error("Embeddings binary not found");
        const buffer = await binRes.arrayBuffer();
        embeddingsRef.current = new Float32Array(buffer);

        // Extract filter option lists
        const stagesSet = new Set<string>();
        const countriesSet = new Set<string>();
        metaData.forEach((item) => {
          if (item["Stage / Funding Tier"])
            stagesSet.add(item["Stage / Funding Tier"]);
          if (item.Country) countriesSet.add(item.Country);
        });
        setAvailableStages(Array.from(stagesSet).sort());
        setAvailableCountries(Array.from(countriesSet).sort());
        setCountryCount(countriesSet.size);

        // Build per-cluster aggregates once — this is what the Market Map renders.
        setClusterStats(buildClusterStats(metaData));

        setLoadingData(false);
        setDataReady(true);
      } catch (err) {
        console.error("Error loading dataset files:", err);
        setError("Couldn't load the startup index. Please refresh to try again.");
        setLoadingData(false);
      }
    }

    loadDataset();
  }, []);

  // Close the modal on Escape for keyboard users
  useEffect(() => {
    if (!isModalOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsModalOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isModalOpen]);

  // Move focus into the dialog when it opens
  useEffect(() => {
    if (isModalOpen) modalRef.current?.focus();
  }, [isModalOpen]);

  // Handle saving API key
  const handleSaveApiKey = (e: React.FormEvent) => {
    e.preventDefault();
    localStorage.setItem("gemini_api_key", userApiKey);
    setHasSavedKey(Boolean(userApiKey.trim()));
    setShowKeyInput(false);
  };

  // Clear API key
  const handleClearApiKey = () => {
    localStorage.removeItem("gemini_api_key");
    setUserApiKey("");
    setHasSavedKey(false);
  };

  // Perform search / matching — embeds the query in-browser, then scores locally
  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;

    const seq = ++searchSeq.current;
    setIsSearching(true);
    setError(null);
    setSelectedIds([]);

    try {
      // Lazily build the embedding model on first search (kept off page load).
      if (!isModelLoaded()) {
        setModelState("loading");
        setModelProgress(0);
      }
      const queryVec = await embedQuery(q, (p: ModelProgress) => {
        if (typeof p.progress === "number") setModelProgress(p.progress);
      });
      setModelState("ready");

      if (seq !== searchSeq.current) return; // a newer search superseded this one

      const matrix = embeddingsRef.current;
      const meta = metadataRef.current;
      if (!matrix || meta.length === 0) {
        throw new Error("The startup index isn't loaded yet — please refresh.");
      }
      if (queryVec.length !== EMBEDDING_DIM) {
        throw new Error("Unexpected embedding size from the model.");
      }

      const { low, high } = SCORE_CALIBRATION;
      const numStartups = meta.length;
      const scoredList: Startup[] = new Array(numStartups);

      // Cosine similarity = dot product, since both sides are unit-normalized.
      for (let i = 0; i < numStartups; i++) {
        let dotProduct = 0;
        const startOffset = i * EMBEDDING_DIM;
        for (let d = 0; d < EMBEDDING_DIM; d++) {
          dotProduct += queryVec[d] * matrix[startOffset + d];
        }

        // Rescale the compressed cosine band to a readable 0–100 (see config.ts).
        const scorePercentage = Math.max(
          0,
          Math.min(100, Math.round(((dotProduct - low) / (high - low)) * 100)),
        );

        scoredList[i] = {
          ...meta[i],
          score: scorePercentage,
          rawScore: dotProduct,
        };
      }

      if (seq !== searchSeq.current) return;
      setAllScoredResults(scoredList);
      setSearchedQuery(q);
      mainRef.current?.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      console.error(err);
      if (seq === searchSeq.current) {
        setError(errMessage(err));
        setModelState(isModelLoaded() ? "ready" : "idle");
      }
    } finally {
      if (seq === searchSeq.current) setIsSearching(false);
    }
  };

  const hasSearched = allScoredResults.length > 0;
  const anyFilterActive = Boolean(
    filterStage || filterCountry || filterHiring || filterMinYear || filterCluster,
  );
  // Three coherent states: the on-load Market Map, a filtered Browse list
  // (no search yet), and the scored Search results.
  const view: "map" | "browse" | "search" = hasSearched
    ? "search"
    : anyFilterActive
      ? "browse"
      : "map";

  const selectedCluster = useMemo(
    () => clusterStats.find((c) => c.name === filterCluster) ?? null,
    [clusterStats, filterCluster],
  );
  const maxClusterSize = clusterStats[0]?.size ?? 1;

  // Jump into a theme from a Market Map card.
  const selectCluster = (name: string) => {
    setFilterCluster(name);
    mainRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  };

  const clearAllFilters = () => {
    setFilterCluster("");
    setFilterStage("");
    setFilterCountry("");
    setFilterHiring("");
    setFilterMinYear("");
  };

  // Apply filters to the active base list (scored results, or the raw corpus
  // when only browsing). Cluster + the classic filters compose together.
  const filteredResults = useMemo(() => {
    const base: Startup[] = hasSearched ? allScoredResults : corpus;
    let list = [...base];

    if (filterCluster) {
      list = list.filter((item) => item.clusterName === filterCluster);
    }
    if (filterStage) {
      list = list.filter((item) => item["Stage / Funding Tier"] === filterStage);
    }
    if (filterCountry) {
      list = list.filter((item) => item.Country === filterCountry);
    }
    if (filterHiring === "hiring") {
      list = list.filter((item) => item["Is Hiring Now"] === true);
    } else if (filterHiring === "not_hiring") {
      list = list.filter((item) => item["Is Hiring Now"] === false);
    }
    if (filterMinYear) {
      const year = parseInt(filterMinYear);
      if (!isNaN(year)) {
        list = list.filter((item) => {
          const founded = parseInt(item["Year Founded"] as string);
          return !isNaN(founded) && founded >= year;
        });
      }
    }

    // Searched → rank by cosine fit. Browsing → alphabetical, stable & neutral.
    return hasSearched
      ? list.sort((a, b) => (b.rawScore || 0) - (a.rawScore || 0))
      : list.sort((a, b) => (a.Name || "").localeCompare(b.Name || ""));
  }, [
    hasSearched,
    allScoredResults,
    corpus,
    filterCluster,
    filterStage,
    filterCountry,
    filterHiring,
    filterMinYear,
  ]);

  // Take top matches for view and stats calculations
  const displayResults = useMemo(() => {
    return filteredResults.slice(0, TOP_K);
  }, [filteredResults]);

  // Compute breakdown charts for top matches (search view)
  const topCountriesChart = useMemo(() => {
    const counts: Record<string, number> = {};
    displayResults.forEach((item) => {
      if (item.Country) {
        counts[item.Country] = (counts[item.Country] || 0) + 1;
      }
    });

    return Object.entries(counts)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 5);
  }, [displayResults]);

  const topStagesChart = useMemo(() => {
    const counts: Record<string, number> = {};
    displayResults.forEach((item) => {
      const stage = item["Stage / Funding Tier"] || "Unknown";
      counts[stage] = (counts[stage] || 0) + 1;
    });

    return Object.entries(counts)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value)
      .slice(0, 5);
  }, [displayResults]);

  // Most common ML theme across the top matches — a sharper headline than the
  // near-constant "hiring rate" (93% of the dataset is flagged open to talent).
  const dominantTheme = useMemo(() => {
    const counts: Record<string, number> = {};
    displayResults.forEach((item) => {
      const d = item.clusterName || "—";
      counts[d] = (counts[d] || 0) + 1;
    });
    const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    return top ? top[0] : "—";
  }, [displayResults]);

  // Handle Multi-Selection
  const handleToggleSelect = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    );
  };

  const handleSelectAllVisible = () => {
    const visibleIds = displayResults.map((r) => r["Startup ID"]).filter(Boolean);
    const allSelected = visibleIds.every((id) => selectedIds.includes(id));
    if (allSelected) {
      setSelectedIds((prev) => prev.filter((id) => !visibleIds.includes(id)));
    } else {
      setSelectedIds((prev) => Array.from(new Set([...prev, ...visibleIds])));
    }
  };

  // Export Salesforce CSV
  const handleExportCSV = () => {
    const listToExport =
      selectedIds.length > 0
        ? displayResults.filter((r) => selectedIds.includes(r["Startup ID"]))
        : displayResults; // Fallback to top matches if none checked

    const headers = [
      "Company Name",
      "Description",
      "Stage",
      "City",
      "Country",
      "Website Search",
      "Is Hiring Now",
      "Match Score %",
      "ML Theme",
      "Primary Domain",
      "Year Founded",
      "Accelerators",
      "Last Funding Amount",
    ];

    const rows = listToExport.map((r) => [
      `"${(r.Name || "").replace(/"/g, '""')}"`,
      `"${(r.Description || "").replace(/"/g, '""')}"`,
      `"${r["Stage / Funding Tier"] || ""}"`,
      `"${r.City || ""}"`,
      `"${r.Country || ""}"`,
      `"${companySearchUrl(r)}"`,
      r["Is Hiring Now"] ? "YES" : "NO",
      typeof r.score === "number" ? `${r.score}%` : "",
      `"${r.clusterName || ""}"`,
      `"${r.Domain || ""}"`,
      r["Year Founded"] || "",
      `"${(r.Accelerators || "").replace(/"/g, '""')}"`,
      r["Last Funding Amount"] || "",
    ]);

    const csvContent = [headers.join(","), ...rows.map((e) => e.join(","))].join(
      "\n",
    );
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8-sig;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute(
      "download",
      `icp_radar_salesforce_leads_${new Date().toISOString().slice(0, 10)}.csv`,
    );
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Open Modal & Generate Challenger Outreach
  const handleOpenOutreachModal = (startup: Startup) => {
    setActiveStartup(startup);
    setOutreachDraft("");
    setIsModalOpen(true);
  };

  const handleGenerateOutreach = async () => {
    if (!activeStartup || !searchedQuery) return;
    setIsGeneratingDraft(true);
    setOutreachDraft("");

    try {
      const res = await fetch("/api/outreach", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          productDescription: searchedQuery,
          startupName: activeStartup.Name,
          startupDescription: activeStartup.Description,
          startupStage: activeStartup["Stage / Funding Tier"],
          startupCountry: activeStartup.Country,
          userApiKey: userApiKey.trim() || undefined,
        }),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        // Hit the shared-key limit → nudge them to add their own key.
        if (res.status === 429) setShowKeyInput(true);
        throw new Error(errorData.error || "Failed to generate outreach email");
      }

      const { outreachDraft: draftText } = await res.json();
      setOutreachDraft(draftText);
    } catch (err) {
      setOutreachDraft(`Error: ${errMessage(err)}`);
    } finally {
      setIsGeneratingDraft(false);
    }
  };

  const handleCopyToClipboard = () => {
    if (!outreachDraft) return;
    navigator.clipboard.writeText(outreachDraft);
    setCopyFeedback(true);
    setTimeout(() => setCopyFeedback(false), 2000);
  };

  // Shared card renderer — works scored (search) or unscored (browse).
  const renderStartupCard = (startup: Startup) => {
    const isChecked = selectedIds.includes(startup["Startup ID"]);
    const scored = typeof startup.score === "number";
    return (
      <article className="startup-card" key={startup["Startup ID"]}>
        <div className="startup-details">
          <div className="startup-header">
            <input
              type="checkbox"
              aria-label={`Select ${startup.Name}`}
              checked={isChecked}
              onChange={() => handleToggleSelect(startup["Startup ID"])}
              style={{
                cursor: "pointer",
                width: "0.95rem",
                height: "0.95rem",
                marginRight: "0.25rem",
              }}
            />
            <h4 className="startup-name">{startup.Name}</h4>
            <span className="badge badge-stage">
              {startup["Stage / Funding Tier"]}
            </span>
            {startup.clusterName && (
              <button
                type="button"
                className="badge badge-cluster"
                onClick={() => selectCluster(startup.clusterName)}
                title={`Filter to the ${startup.clusterName} theme`}
              >
                {startup.clusterName}
              </button>
            )}
            {startup["Is Hiring Now"] ? (
              <span className="badge badge-hiring">Hiring</span>
            ) : (
              <span className="badge badge-not-hiring">Closed</span>
            )}
          </div>

          <p className="startup-description">{startup.Description}</p>

          <div className="startup-meta-row">
            <div className="meta-item">
              📍{" "}
              <span>
                {startup.City ? `${startup.City}, ` : ""}
                {startup.Country}
              </span>
            </div>
            {startup.Domain && (
              <div className="meta-item">
                🏷️ <span>{startup.Domain}</span>
              </div>
            )}
            {startup["Year Founded"] && (
              <div className="meta-item">
                📅 <span>Est. {startup["Year Founded"]}</span>
              </div>
            )}
            {startup.Accelerators && (
              <div className="meta-item">
                ⚡ <span>{startup.Accelerators}</span>
              </div>
            )}
          </div>
        </div>

        <div className="startup-action-panel">
          {scored ? (
            <div className="score-display">
              <span className="score-value">{startup.score}%</span>
              <span className="score-label">ICP Fit</span>
            </div>
          ) : (
            <div className="score-display">
              <span className="score-value" style={{ fontSize: "1.1rem", color: "var(--text-secondary)" }}>
                {startup["Stage / Funding Tier"]}
              </span>
              <span className="score-label">Run a search to score</span>
            </div>
          )}

          <div className="action-buttons">
            <a
              href={companySearchUrl(startup)}
              target="_blank"
              rel="noopener noreferrer"
              className="btn-secondary"
              style={{
                padding: "0.45rem",
                fontSize: "0.8rem",
                width: "100%",
                textDecoration: "none",
                textAlign: "center",
              }}
            >
              Find site ↗
            </a>
            {scored && (
              <button
                onClick={() => handleOpenOutreachModal(startup)}
                className="btn-primary"
                style={{
                  padding: "0.45rem",
                  fontSize: "0.8rem",
                  width: "100%",
                  background:
                    "linear-gradient(135deg, var(--accent-purple), #7c3aed)",
                }}
              >
                Challenger SDR
              </button>
            )}
          </div>
        </div>
      </article>
    );
  };

  return (
    <div className="app-container">
      {/* Sidebar Control Console */}
      <aside className="sidebar">
        <div className="logo-section">
          <div className="logo-icon">🛰️</div>
          <div>
            <h1 className="logo-title">ICP Radar</h1>
            <p className="logo-subtitle">Sales Intelligence</p>
          </div>
        </div>

        {/* API Key configuration banner */}
        <div className="apikey-banner">
          <span>{hasSavedKey ? "🔑 Gemini key active" : "⚠️ Gemini key optional"}</span>
          <button onClick={() => setShowKeyInput(!showKeyInput)}>
            {showKeyInput ? "Close" : "Configure"}
          </button>
        </div>

        {showKeyInput && (
          <form
            onSubmit={handleSaveApiKey}
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "0.5rem",
              background: "rgba(255,255,255,0.03)",
              padding: "0.75rem",
              borderRadius: "8px",
              border: "1px solid var(--border-color)",
            }}
          >
            <label htmlFor="gemini-key" className="form-label">
              Gemini API Key
            </label>
            <input
              id="gemini-key"
              type="password"
              className="text-input"
              placeholder="AI Studio Key"
              value={userApiKey}
              onChange={(e) => setUserApiKey(e.target.value)}
            />
            <div style={{ display: "flex", gap: "0.5rem", marginTop: "0.25rem" }}>
              <button
                type="submit"
                className="btn-primary"
                style={{ padding: "0.4rem", fontSize: "0.8rem" }}
              >
                Save
              </button>
              {hasSavedKey && (
                <button
                  type="button"
                  className="btn-secondary"
                  style={{ padding: "0.4rem", fontSize: "0.8rem" }}
                  onClick={handleClearApiKey}
                >
                  Clear
                </button>
              )}
            </div>
          </form>
        )}

        {/* Query Input Box */}
        <form onSubmit={handleSearch} className="search-form">
          <div className="form-group">
            <label className="form-label" htmlFor="query-input">
              Product Value Proposition
            </label>
            <textarea
              id="query-input"
              className="textarea-input"
              placeholder="e.g. We sell automated data auditing software for heavy industry and logistics SMEs to verify invoice lines..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={loadingData || isSearching}
              required
            />
          </div>
          <button
            type="submit"
            className="btn-primary"
            disabled={loadingData || isSearching || !query.trim()}
          >
            {isSearching ? (
              <>
                <div className="spinner"></div>
                {modelState === "loading" ? "Loading model…" : "Matching…"}
              </>
            ) : (
              <>Match ideal ICP</>
            )}
          </button>

          {modelState === "loading" && (
            <div className="model-load">
              <span>
                Loading search model (~23 MB, one-time)
                {modelProgress > 0 ? ` · ${Math.round(modelProgress)}%` : "…"}
              </span>
              <div className="model-load-bar">
                <div
                  className="model-load-fill"
                  style={{ width: `${Math.max(5, modelProgress)}%` }}
                ></div>
              </div>
            </div>
          )}
        </form>

        <hr
          style={{
            border: "none",
            borderTop: "1px solid var(--border-color)",
            margin: "0.5rem 0",
          }}
        />

        {/* Parameters Console */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <div className="flex-row-center" style={{ justifyContent: "space-between" }}>
            <h3 className="form-label" style={{ color: "#fff", fontSize: "0.8rem" }}>
              Segmentation Filters
            </h3>
            {anyFilterActive && (
              <button
                type="button"
                onClick={clearAllFilters}
                style={{
                  background: "none",
                  border: "none",
                  color: "var(--accent-blue)",
                  fontSize: "0.75rem",
                  cursor: "pointer",
                  padding: 0,
                }}
              >
                Reset
              </button>
            )}
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="filter-cluster">
              Semantic Theme (ML Cluster)
            </label>
            <select
              id="filter-cluster"
              className="select-input"
              value={filterCluster}
              onChange={(e) => setFilterCluster(e.target.value)}
              disabled={!dataReady}
            >
              <option value="">All 15 Themes</option>
              {clusterStats.map((c) => (
                <option key={c.id} value={c.name}>
                  {c.name} · {c.size}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="filter-stage">
              Funding Stage
            </label>
            <select
              id="filter-stage"
              className="select-input"
              value={filterStage}
              onChange={(e) => setFilterStage(e.target.value)}
              disabled={!dataReady}
            >
              <option value="">All Stages</option>
              {availableStages.map((stage) => (
                <option key={stage} value={stage}>
                  {stage}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="filter-country">
              Geographic Country
            </label>
            <select
              id="filter-country"
              className="select-input"
              value={filterCountry}
              onChange={(e) => setFilterCountry(e.target.value)}
              disabled={!dataReady}
            >
              <option value="">All Countries</option>
              {availableCountries.map((country) => (
                <option key={country} value={country}>
                  {country}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="filter-hiring">
              Hiring Status
            </label>
            <select
              id="filter-hiring"
              className="select-input"
              value={filterHiring}
              onChange={(e) => setFilterHiring(e.target.value)}
              disabled={!dataReady}
            >
              <option value="">All Hiring Statuses</option>
              <option value="hiring">Actively Hiring Now</option>
              <option value="not_hiring">Not Hiring</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="filter-year">
              Founded On or After Year
            </label>
            <input
              id="filter-year"
              type="number"
              className="text-input"
              placeholder="e.g. 2020"
              value={filterMinYear}
              onChange={(e) => setFilterMinYear(e.target.value)}
              disabled={!dataReady}
            />
          </div>
        </div>
      </aside>

      {/* Main Panel */}
      <main className="main-content" ref={mainRef}>
        {/* Header section */}
        <header className="dashboard-header">
          <div className="header-text">
            <h2>European Startup ICP Radar</h2>
            <p>Target Account Intelligence Matrix</p>
          </div>
          <div
            style={{
              color: "var(--text-muted)",
              fontSize: "0.85rem",
              textAlign: "right",
            }}
          >
            {loadingData
              ? "Syncing data layers..."
              : `Indexed ${totalStartupsCount} European startups`}
          </div>
        </header>

        {/* Global Error Banner */}
        {error && (
          <div
            style={{
              padding: "1rem",
              background: "rgba(239, 68, 68, 0.1)",
              border: "1px solid rgba(239, 68, 68, 0.25)",
              borderRadius: "8px",
              color: "#f87171",
              fontSize: "0.9rem",
            }}
          >
            {error}
          </div>
        )}

        {loadingData ? (
          /* Database Loading screen */
          <div className="empty-state">
            <div className="spinner" style={{ width: "40px", height: "40px" }}></div>
            <h3 className="empty-state-title" style={{ marginTop: "1rem" }}>
              Ingesting Database Layers
            </h3>
            <p style={{ maxWidth: "380px" }}>
              Loading 3,756 startup profiles and the precomputed vector matrix into
              memory for high-velocity local execution.
            </p>
          </div>
        ) : (
          <>
            {/* Selected-cluster drill-down — shows in browse + search views */}
            {selectedCluster && (view === "browse" || view === "search") && (
              <ClusterInsights
                cluster={selectedCluster}
                onClear={() => setFilterCluster("")}
              />
            )}

            {/* ---------- MARKET MAP (on load, no search, no filters) ---------- */}
            {view === "map" && (
              <>
                <section className="stats-grid">
                  <div className="stat-card">
                    <span className="stat-card-label">Companies Indexed</span>
                    <span className="stat-card-value">{totalStartupsCount}</span>
                    <span className="stat-card-desc">
                      European startups, embedded &amp; clustered
                    </span>
                  </div>
                  <div className="stat-card">
                    <span className="stat-card-label">ML-Discovered Themes</span>
                    <span className="stat-card-value">{clusterStats.length}</span>
                    <span className="stat-card-desc">
                      k-means over 384-dim MiniLM vectors
                    </span>
                  </div>
                  <div className="stat-card">
                    <span className="stat-card-label">Countries Covered</span>
                    <span className="stat-card-value">{countryCount}</span>
                    <span className="stat-card-desc">
                      across the European startup map
                    </span>
                  </div>
                </section>

                <div className="thesis-banner">
                  <strong>How to read this map.</strong> These {clusterStats.length}{" "}
                  themes were discovered by an unsupervised model clustering the{" "}
                  <em>language</em> of {totalStartupsCount} company descriptions — not
                  their self-filed categories. Thirteen themes rebuild the stated
                  taxonomy (a validation result); two —{" "}
                  <em>Technical Workflow Layer</em> and <em>Commerce Operations</em> —
                  merge several filed categories into one behaviour the market actually
                  exhibits. Click any theme to drill into its maturity, geography and
                  composition.
                </div>

                <section className="results-header" style={{ marginBottom: "-0.5rem" }}>
                  <span className="results-count">
                    The European startup landscape, in {clusterStats.length} themes
                  </span>
                </section>

                <section className="market-map-grid">
                  {clusterStats.map((c) => {
                    const buckets = bucketStages(c.stageCounts);
                    return (
                      <button
                        type="button"
                        className="cluster-card"
                        key={c.id}
                        onClick={() => selectCluster(c.name)}
                      >
                        <div className="cluster-card-head">
                          <span className="cluster-card-name">{c.name}</span>
                          <span className="cluster-card-size">{c.size}</span>
                        </div>

                        <div className="cluster-size-bar">
                          <div
                            className="cluster-size-fill"
                            style={{
                              width: `${(c.size / maxClusterSize) * 100}%`,
                            }}
                          ></div>
                        </div>

                        <p className="cluster-tagline">
                          {CLUSTER_TAGLINES[c.name] ?? ""}
                        </p>

                        <div className="maturity-bar mini" aria-hidden="true">
                          {buckets.map(
                            (b) =>
                              b.pct > 0 && (
                                <div
                                  key={b.label}
                                  className="maturity-seg"
                                  style={{ width: `${b.pct}%`, background: b.color }}
                                  title={`${b.label}: ${b.pct}%`}
                                ></div>
                              ),
                          )}
                        </div>

                        <div className="cluster-card-foot">
                          <span>{c.sharePct.toFixed(1)}% of corpus</span>
                          {c.topCountries[0] && (
                            <span>🌍 {c.topCountries[0].name}</span>
                          )}
                          {c.crossTaxonomy && (
                            <span className="cross-tax-dot" title="Cross-taxonomy theme">
                              ◆ {c.distinctDomains} categories
                            </span>
                          )}
                        </div>
                      </button>
                    );
                  })}
                </section>
              </>
            )}

            {/* ---------- BROWSE (filters applied, no search yet) ---------- */}
            {view === "browse" && (
              <section className="results-section">
                <div className="results-header">
                  <div className="flex-row-center">
                    <button
                      type="button"
                      className="btn-secondary"
                      style={{ width: "auto", padding: "0.4rem 0.85rem", fontSize: "0.8rem" }}
                      onClick={clearAllFilters}
                    >
                      ← Market Map
                    </button>
                    <span className="results-count">
                      {filteredResults.length} companies match — browsing unscored.
                      Run a value-prop search to rank them.
                    </span>
                  </div>
                  <button
                    onClick={handleExportCSV}
                    className="btn-secondary"
                    style={{ width: "auto", padding: "0.45rem 1rem", fontSize: "0.85rem" }}
                  >
                    📦 Export ({selectedIds.length || displayResults.length})
                  </button>
                </div>

                {displayResults.length === 0 ? (
                  <div className="empty-state" style={{ padding: "3rem 2rem" }}>
                    <h3 className="empty-state-title">No companies match these filters</h3>
                    <p>Try easing a filter or clearing the theme.</p>
                  </div>
                ) : (
                  <div className="results-list">
                    {displayResults.map((s) => renderStartupCard(s))}
                  </div>
                )}
                {filteredResults.length > displayResults.length && (
                  <p style={{ color: "var(--text-muted)", fontSize: "0.8rem", textAlign: "center" }}>
                    Showing the first {displayResults.length} of {filteredResults.length}.
                    Narrow the filters or run a search to surface the best fits.
                  </p>
                )}
              </section>
            )}

            {/* ---------- SEARCH RESULTS ---------- */}
            {view === "search" && (
              <>
                {/* Stats row */}
                <section className="stats-grid">
                  <div className="stat-card">
                    <span className="stat-card-label">Companies in Scope</span>
                    <span className="stat-card-value">{filteredResults.length}</span>
                    <span className="stat-card-desc">
                      Pass your active filters (top {TOP_K} scored below)
                    </span>
                  </div>
                  <div className="stat-card">
                    <span className="stat-card-label">Mean Match Score</span>
                    <span className="stat-card-value">
                      {Math.round(
                        displayResults.reduce((acc, curr) => acc + (curr.score || 0), 0) /
                          (displayResults.length || 1),
                      )}
                      %
                    </span>
                    <span className="stat-card-desc">
                      Average semantic fit of displayed matches
                    </span>
                  </div>
                  <div className="stat-card">
                    <span className="stat-card-label">Dominant Theme</span>
                    <span className="stat-card-value" style={{ fontSize: "1.25rem" }}>
                      {dominantTheme}
                    </span>
                    <span className="stat-card-desc">
                      Most common ML theme across top matches
                    </span>
                  </div>
                </section>

                {/* Custom CSS Charts grid */}
                <section className="analytics-section">
                  {/* Country Density Card */}
                  <div className="analytics-card">
                    <h3 className="analytics-title">
                      Top Matching Countries (Top {TOP_K} Matches)
                    </h3>
                    <div className="chart-bar-container">
                      {topCountriesChart.map((row) => {
                        const maxVal = topCountriesChart[0]?.value || 1;
                        const percentWidth = (row.value / maxVal) * 100;
                        return (
                          <div className="chart-row" key={row.name}>
                            <span className="chart-label">{row.name}</span>
                            <div className="chart-bar-wrapper">
                              <div
                                className="chart-bar-fill"
                                style={{
                                  width: `${percentWidth}%`,
                                  background:
                                    "linear-gradient(90deg, var(--accent-blue), var(--accent-purple))",
                                }}
                              ></div>
                            </div>
                            <span className="chart-val">{row.value}</span>
                          </div>
                        );
                      })}
                      {topCountriesChart.length === 0 && (
                        <div
                          style={{
                            color: "var(--text-muted)",
                            fontSize: "0.85rem",
                            textAlign: "center",
                            padding: "1rem",
                          }}
                        >
                          No countries represented
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Funding Stage Density Card */}
                  <div className="analytics-card">
                    <h3 className="analytics-title">
                      Funding Stage Profile (Top {TOP_K} Matches)
                    </h3>
                    <div className="chart-bar-container">
                      {topStagesChart.map((row) => {
                        const maxVal = topStagesChart[0]?.value || 1;
                        const percentWidth = (row.value / maxVal) * 100;
                        return (
                          <div className="chart-row" key={row.name}>
                            <span className="chart-label">{row.name}</span>
                            <div className="chart-bar-wrapper">
                              <div
                                className="chart-bar-fill"
                                style={{
                                  width: `${percentWidth}%`,
                                  background:
                                    "linear-gradient(90deg, var(--accent-green), var(--accent-blue))",
                                }}
                              ></div>
                            </div>
                            <span className="chart-val">{row.value}</span>
                          </div>
                        );
                      })}
                      {topStagesChart.length === 0 && (
                        <div
                          style={{
                            color: "var(--text-muted)",
                            fontSize: "0.85rem",
                            textAlign: "center",
                            padding: "1rem",
                          }}
                        >
                          No stages represented
                        </div>
                      )}
                    </div>
                  </div>
                </section>

                {/* Results Grid List */}
                <section className="results-section">
                  <div className="results-header">
                    <div className="flex-row-center">
                      <input
                        type="checkbox"
                        id="select-all-box"
                        style={{ cursor: "pointer", width: "1rem", height: "1rem" }}
                        checked={
                          displayResults.length > 0 &&
                          displayResults.every((r) =>
                            selectedIds.includes(r["Startup ID"]),
                          )
                        }
                        onChange={handleSelectAllVisible}
                      />
                      <label
                        htmlFor="select-all-box"
                        className="results-count"
                        style={{ cursor: "pointer" }}
                      >
                        Showing top {displayResults.length} matches ({selectedIds.length}{" "}
                        selected)
                      </label>
                    </div>
                    <div style={{ display: "flex", gap: "0.75rem" }}>
                      <button
                        onClick={handleExportCSV}
                        className="btn-secondary"
                        style={{ padding: "0.45rem 1rem", fontSize: "0.85rem" }}
                      >
                        📦 Export leads for CRM ({selectedIds.length || displayResults.length})
                      </button>
                    </div>
                  </div>

                  {displayResults.length === 0 ? (
                    <div className="empty-state" style={{ padding: "3rem 2rem" }}>
                      <h3 className="empty-state-title">
                        No startups match active filters
                      </h3>
                      <p>Try easing your filtering parameters or min year constraints.</p>
                    </div>
                  ) : (
                    <div className="results-list">
                      {displayResults.map((s) => renderStartupCard(s))}
                    </div>
                  )}
                </section>
              </>
            )}
          </>
        )}
      </main>

      {/* Outreach Generation Modal Overlay */}
      <div
        className={`modal-overlay ${isModalOpen ? "active" : ""}`}
        onClick={() => setIsModalOpen(false)}
        aria-hidden={!isModalOpen}
        inert={!isModalOpen}
      >
        <div
          className="modal-content"
          onClick={(e) => e.stopPropagation()}
          role="dialog"
          aria-modal="true"
          aria-labelledby="outreach-modal-title"
          tabIndex={-1}
          ref={modalRef}
        >
          <div className="modal-header">
            <h3 className="modal-title" id="outreach-modal-title">
              Challenger Sales Script — {activeStartup?.Name}
            </h3>
            <button
              className="modal-close"
              onClick={() => setIsModalOpen(false)}
              aria-label="Close dialog"
            >
              &times;
            </button>
          </div>

          <div className="modal-body">
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
              Generates a tailored SDR outreach email specifically leveraging the
              5-step Challenger Methodology based on the company&apos;s profile.
            </p>

            <div
              style={{
                background: "rgba(255,255,255,0.02)",
                padding: "1rem",
                border: "1px solid var(--border-color)",
                borderRadius: "8px",
                fontSize: "0.8rem",
                display: "flex",
                flexDirection: "column",
                gap: "0.5rem",
              }}
            >
              <div>
                <strong>Target:</strong> {activeStartup?.Name} (
                {activeStartup?.["Stage / Funding Tier"]})
              </div>
              <div>
                <strong>Value Proposition Hook:</strong> {searchedQuery}
              </div>
            </div>

            {outreachDraft ? (
              <div className="outreach-draft-container">{outreachDraft}</div>
            ) : (
              <div className="empty-state" style={{ padding: "2rem 1rem" }}>
                <p>
                  Generate a draft with Gemini. A shared key powers the demo (rate
                  limited) — add your own in the sidebar to bypass the limit.
                </p>
              </div>
            )}
          </div>

          <div className="modal-footer">
            <button className="btn-secondary" onClick={() => setIsModalOpen(false)}>
              Close
            </button>
            {outreachDraft && (
              <button
                className="btn-secondary"
                onClick={handleCopyToClipboard}
                style={{
                  background: "rgba(16, 185, 129, 0.1)",
                  borderColor: "rgba(16, 185, 129, 0.2)",
                  color: "#34d399",
                }}
              >
                {copyFeedback ? "✓ Copied" : "📋 Copy Script"}
              </button>
            )}
            <button
              className="btn-primary"
              disabled={isGeneratingDraft}
              onClick={handleGenerateOutreach}
            >
              {isGeneratingDraft ? (
                <>
                  <div className="spinner"></div>
                  Generating...
                </>
              ) : (
                "Generate Draft"
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Build per-cluster aggregates from the enriched dataset (run once on load).
function buildClusterStats(rows: Startup[]): ClusterStat[] {
  const total = rows.length || 1;
  const acc = new Map<
    number,
    {
      id: number;
      name: string;
      size: number;
      stage: Record<string, number>;
      country: Record<string, number>;
      domain: Record<string, number>;
    }
  >();

  for (const r of rows) {
    const id = r.clusterId ?? -1;
    let s = acc.get(id);
    if (!s) {
      s = { id, name: r.clusterName ?? "Unclustered", size: 0, stage: {}, country: {}, domain: {} };
      acc.set(id, s);
    }
    s.size += 1;
    const stage = r["Stage / Funding Tier"] || "Unknown";
    const country = r.Country || "Unknown";
    const domain = r.Domain || "Unknown";
    s.stage[stage] = (s.stage[stage] || 0) + 1;
    s.country[country] = (s.country[country] || 0) + 1;
    s.domain[domain] = (s.domain[domain] || 0) + 1;
  }

  return Array.from(acc.values())
    .sort((a, b) => a.id - b.id)
    .map((s) => {
      const distinctDomains = Object.values(s.domain).filter(
        (v) => v / s.size >= 0.1,
      ).length;
      return {
        id: s.id,
        name: s.name,
        size: s.size,
        sharePct: (s.size / total) * 100,
        stageCounts: s.stage,
        topCountries: topEntries(s.country, 3),
        topDomains: topEntries(s.domain, 5),
        distinctDomains,
        crossTaxonomy: distinctDomains >= 3,
      };
    });
}

// Drill-down panel for a single selected theme: maturity, geography, composition.
function ClusterInsights({
  cluster,
  onClear,
}: {
  cluster: ClusterStat;
  onClear: () => void;
}) {
  const buckets = bucketStages(cluster.stageCounts).filter((b) => b.value > 0);
  const maxCountry = cluster.topCountries[0]?.value || 1;
  const maxDomain = cluster.topDomains[0]?.value || 1;

  return (
    <section className="insights-panel">
      <div className="insights-head">
        <div>
          <div className="flex-row-center" style={{ gap: "0.75rem", flexWrap: "wrap" }}>
            <h3 className="insights-title">{cluster.name}</h3>
            <span className="badge badge-stage">{cluster.size} companies</span>
            <span className="badge badge-stage">
              {cluster.sharePct.toFixed(1)}% of corpus
            </span>
            {cluster.crossTaxonomy && (
              <span className="badge badge-cross">
                ◆ Cross-taxonomy: {cluster.distinctDomains} filed categories merged
              </span>
            )}
          </div>
          <p className="insights-tagline">{CLUSTER_TAGLINES[cluster.name] ?? ""}</p>
        </div>
        <button
          type="button"
          className="btn-secondary"
          style={{ width: "auto", padding: "0.4rem 0.85rem", fontSize: "0.8rem" }}
          onClick={onClear}
        >
          Clear theme
        </button>
      </div>

      <div className="insights-grid">
        {/* Maturity profile — stacked bar */}
        <div className="insight-card">
          <h4 className="insight-card-title">Maturity Profile</h4>
          <div className="maturity-bar">
            {buckets.map((b) => (
              <div
                key={b.label}
                className="maturity-seg"
                style={{ width: `${b.pct}%`, background: b.color }}
                title={`${b.label}: ${b.value} (${b.pct}%)`}
              ></div>
            ))}
          </div>
          <div className="maturity-legend">
            {buckets.map((b) => (
              <div className="legend-item" key={b.label}>
                <span className="legend-dot" style={{ background: b.color }}></span>
                <span className="legend-label">{b.label}</span>
                <span className="legend-val">{b.pct}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* Geographic hubs */}
        <div className="insight-card">
          <h4 className="insight-card-title">Geographic Hubs</h4>
          <div className="chart-bar-container">
            {cluster.topCountries.map((row) => (
              <div className="chart-row" key={row.name}>
                <span className="chart-label">{row.name}</span>
                <div className="chart-bar-wrapper">
                  <div
                    className="chart-bar-fill"
                    style={{
                      width: `${(row.value / maxCountry) * 100}%`,
                      background:
                        "linear-gradient(90deg, var(--accent-blue), var(--accent-purple))",
                    }}
                  ></div>
                </div>
                <span className="chart-val">{row.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Composition — dominant source categories */}
        <div className="insight-card">
          <h4 className="insight-card-title">
            Composition <span className="insight-sub">(filed categories)</span>
          </h4>
          <div className="chart-bar-container">
            {cluster.topDomains.map((row) => (
              <div className="chart-row" key={row.name}>
                <span className="chart-label" title={row.name}>
                  {row.name}
                </span>
                <div className="chart-bar-wrapper">
                  <div
                    className="chart-bar-fill"
                    style={{
                      width: `${(row.value / maxDomain) * 100}%`,
                      background:
                        "linear-gradient(90deg, var(--accent-green), var(--accent-blue))",
                    }}
                  ></div>
                </div>
                <span className="chart-val">{row.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
