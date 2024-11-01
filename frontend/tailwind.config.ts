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
        primaryBlue: "#0D1127",
        secondaryBlue: "#0C3C60",
        tertiaryBlue: "#B0D0E8",
        fadedGold: "#DEC328",
      },
      fontFamily: {
        sans: ['circular', 'sans-serif'],
        circular: ['circular', 'sans-serif'],
        planet: ['planet', 'sans-serif']
      },
    },
  },
  plugins: [],
};
export default config;
