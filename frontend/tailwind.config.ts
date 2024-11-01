import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
      },
      fontFamily: {
        sans: [
          "CircularStd-Book",
          "CircularStd-Medium",
          "CircularStd-Bold",
          "sans-serif"
        ],
        circular: [
          "CircularStd-Book",
          "CircularStd-Medium",
          "CircularStd-Bold",
          "sans-serif"
        ]
      },
    },
  },
  plugins: [],
};
export default config;
