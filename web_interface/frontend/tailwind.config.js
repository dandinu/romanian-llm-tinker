/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        border: "oklch(92.2% 0 0)",
        input: "oklch(92.2% 0 0)",
        ring: "oklch(64.8% 0.169 240.3)",
        background: "#fff",
        foreground: "oklch(14.5% 0 0)",
        primary: {
          DEFAULT: "#0B99FF",
          foreground: "#fff",
        },
        secondary: {
          DEFAULT: "#f97316",
          foreground: "#fff",
        },
        destructive: {
          DEFAULT: "#F45757",
          foreground: "#fff",
        },
        success: {
          DEFAULT: "#10b981",
          foreground: "#fff",
        },
        warning: {
          DEFAULT: "#f59e0b",
          foreground: "#fff",
        },
        muted: {
          DEFAULT: "oklch(97% 0 0)",
          foreground: "oklch(45.8% 0.004 258.3)",
        },
        accent: {
          DEFAULT: "oklch(97% 0 0)",
          foreground: "oklch(14.5% 0 0)",
        },
        card: {
          DEFAULT: "#fff",
          foreground: "oklch(14.5% 0 0)",
        },
      },
      borderRadius: {
        lg: "10px",
        md: "8px",
        sm: "6px",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "SF Pro Text", "sans-serif"],
      },
      fontSize: {
        xs: "12px",
        sm: "14px",
        base: "16px",
        lg: "18px",
        xl: "20px",
        "2xl": "24px",
        "3xl": "30px",
        "4xl": "36px",
      },
      spacing: {
        sidebar: "256px",
        "sidebar-collapsed": "48px",
      },
      transitionDuration: {
        DEFAULT: "150ms",
      },
      boxShadow: {
        sm: "0 1px 2px 0 rgb(0 0 0 / 0.05)",
        DEFAULT: "0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1)",
        md: "0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  plugins: [],
}
