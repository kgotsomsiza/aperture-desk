import { cloudflareTest } from "@cloudflare/vitest-plugin";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [
    cloudflareTest({
      wrangler: { configPath: "./wrangler.jsonc" },
      miniflare: {
        bindings: { PUBLISH_TOKEN: "unit-test-publish-token" },
        kvNamespaces: ["SNAPSHOTS"],
      },
    }),
  ],
});
