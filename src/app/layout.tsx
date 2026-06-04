import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ICP Radar | B2B Sales Intelligence & Semantic ICP Search",
  description:
    "Identify Ideal Customer Profiles (ICP) across 3,756 European startups using semantic search and precomputed vector embeddings, and generate tailored Challenger Sales outreach.",
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
