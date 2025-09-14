import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  // Enable source maps for better debugging
  productionBrowserSourceMaps: false,
  // Enable webpack source maps in development
  webpack: (config, { dev, isServer }) => {
    if (dev && !isServer) {
      config.devtool = 'eval-source-map';
    }
    return config;
  },
  // Expose environment variables to the client
  env: {
    NEXT_PUBLIC_VERBOSE: process.env.NEXT_VERBOSE,
    NEXT_PUBLIC_DEBUG: process.env.NEXT_DEBUG,
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL,
  },
  // Enable debugging
  experimental: {
    // Enable better debugging support
  },
  // Ensure proper source map generation
  typescript: {
    // Don't ignore TypeScript errors during build for debugging
    ignoreBuildErrors: false,
  },
  eslint: {
    // Don't ignore ESLint errors during build for debugging
    ignoreDuringBuilds: false,
  },
};

export default nextConfig;
