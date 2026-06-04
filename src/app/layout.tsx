import type { Metadata } from "next";
import "./globals.css";

const SITE_URL = "https://icp-radar.vercel.app";
const OG_DESCRIPTION =
  "Identify Ideal Customer Profiles (ICP) across 3,756 European startups using semantic search and precomputed vector embeddings, and generate tailored Challenger Sales outreach.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: "ICP Radar | B2B Sales Intelligence & Semantic ICP Search",
  description: OG_DESCRIPTION,
  openGraph: {
    type: "website",
    url: SITE_URL,
    title: "ICP Radar — Semantic Target-Account Discovery",
    description: OG_DESCRIPTION,
    siteName: "ICP Radar",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "ICP Radar — European startup ICP search and intelligence dashboard",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "ICP Radar — Semantic Target-Account Discovery",
    description: OG_DESCRIPTION,
    images: ["/og-image.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
