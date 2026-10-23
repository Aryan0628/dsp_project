import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Self-contained server bundle for the Docker image; Vercel ignores this setting.
  output: "standalone",
};

export default nextConfig;
