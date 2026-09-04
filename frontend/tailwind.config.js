/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        background: "#10131a",
        surface: {
          DEFAULT: "#10131a",
          dim: "#10131a",
          bright: "#363941",
          lowest: "#0b0e15",
          low: "#191b23",
          "container-lowest": "#0b0e15",
          "container-low": "#161921",
          container: "#1d2027",
          "container-high": "#272a31",
          "container-highest": "#32353c",
          high: "#272a31",
          highest: "#32353c",
        },
        primary: {
          DEFAULT: "#adc6ff",
          container: "#4d8eff",
          fixed: "#d8e2ff",
          "fixed-dim": "#adc6ff",
        },
        on: {
          primary: "#002e6a",
          "primary-container": "#00285d",
          surface: "#e1e2ec",
          "surface-variant": "#c2c6d6",
          background: "#e1e2ec",
          secondary: "#3c0091",
          "secondary-container": "#c4abff",
          tertiary: "#003640",
          "tertiary-container": "#002f38",
        },
        secondary: {
          DEFAULT: "#d0bcff",
          container: "#571bc1",
          fixed: "#e9ddff",
          "fixed-dim": "#d0bcff",
        },
        tertiary: {
          DEFAULT: "#4cd7f6",
          container: "#009eb9",
          fixed: "#acedff",
          "fixed-dim": "#4cd7f6",
        },
        error: {
          DEFAULT: "#ffb4ab",
          container: "#93000a",
        },
        outline: {
          DEFAULT: "#8c909f",
          variant: "#424754",
        },
        inverse: {
          surface: "#e1e2ec",
          "on-surface": "#2e3038",
          primary: "#005ac2",
        },
      },
      borderRadius: {
        none: "0",
        sm: "0.25rem",
        DEFAULT: "0.375rem",
        md: "0.5rem",
        lg: "0.75rem",
        xl: "1rem",
        "2xl": "1.25rem",
        "3xl": "1.5rem",
        full: "9999px",
      },
      spacing: {
        xs: "8px",
        sm: "16px",
        md: "24px",
        lg: "40px",
        xl: "64px",
        gutter: "24px",
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
        display: ["Space Grotesk", "sans-serif"],
      },
      fontSize: {
        "display-lg": [
          "48px",
          { lineHeight: "1.1", letterSpacing: "-0.02em", fontWeight: "700" },
        ],
        "headline-lg": [
          "32px",
          { lineHeight: "1.2", fontWeight: "600" },
        ],
        "headline-md": [
          "24px",
          { lineHeight: "32px", fontWeight: "600" },
        ],
        "headline-sm": [
          "20px",
          { lineHeight: "28px", fontWeight: "500" },
        ],
        "body-lg": [
          "16px",
          { lineHeight: "24px", fontWeight: "400" },
        ],
        "body-md": [
          "14px",
          { lineHeight: "20px", fontWeight: "400" },
        ],
        "body-sm": [
          "12px",
          { lineHeight: "16px", fontWeight: "400" },
        ],
        "label-lg": [
          "14px",
          { lineHeight: "20px", letterSpacing: "0.01em", fontWeight: "500" },
        ],
        "label-md": [
          "12px",
          { lineHeight: "16px", letterSpacing: "0.05em", fontWeight: "500" },
        ],
        "label-sm": [
          "11px",
          { lineHeight: "16px", letterSpacing: "0.05em", fontWeight: "500" },
        ],
        "code-sm": [
          "12px",
          { lineHeight: "18px", fontWeight: "400" },
        ],
      },
      boxShadow: {
        glow: "0 0 20px rgba(173, 198, 255, 0.15)",
        "glow-lg": "0 0 40px rgba(173, 198, 255, 0.2)",
        "glow-primary": "0 0 20px rgba(173, 198, 255, 0.25)",
        "inner-glow": "inset 0 1px 0 rgba(255,255,255,0.05)",
      },
      animation: {
        "fade-in": "fadeIn 0.5s ease-out",
        "slide-up": "slideUp 0.3s ease-out",
        "slide-in-left": "slideInLeft 0.3s ease-out",
        "pulse-slow": "pulse 3s ease-in-out infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideInLeft: {
          "0%": { opacity: "0", transform: "translateX(-10px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
      },
    },
  },
  plugins: [],
};
