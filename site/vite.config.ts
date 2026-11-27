import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Served from https://vinodanbalagan.github.io/trust-horizon/
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { assetsInlineLimit: 0, chunkSizeWarningLimit: 900 },
});
