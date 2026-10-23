import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Self-contained server bundle for the Docker image only. Vercel packages the app its
  // own way and its build step fails on standalone output, so skip it there (Vercel sets
  // VERCEL=1 during builds).
  output: process.env.VERCEL ? undefined : "standalone",
};

export default nextConfig;
