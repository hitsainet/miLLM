/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Primary accent colors (cyan theme from mockup)
        primary: {
          50: '#ecfeff',
          100: '#cffafe',
          200: '#a5f3fc',
          300: '#67e8f9',
          400: '#22d3ee',
          500: '#06b6d4',
          600: '#0891b2',
          700: '#0e7490',
          800: '#155e75',
          900: '#164e63',
          950: '#083344',
        },
        // Status colors
        success: {
          DEFAULT: '#4ade80',
          light: 'rgba(34, 197, 94, 0.15)',
          border: 'rgba(34, 197, 94, 0.3)',
        },
        warning: {
          DEFAULT: '#fbbf24',
          light: 'rgba(251, 191, 36, 0.15)',
          border: 'rgba(251, 191, 36, 0.3)',
        },
        danger: {
          DEFAULT: '#f87171',
          light: 'rgba(239, 68, 68, 0.15)',
          border: 'rgba(239, 68, 68, 0.3)',
        },
        // Purple for SAE/attached status
        purple: {
          DEFAULT: '#c084fc',
          light: 'rgba(168, 85, 247, 0.15)',
          border: 'rgba(168, 85, 247, 0.3)',
        },
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      backgroundColor: {
        'app': 'linear-gradient(135deg, #0a0f1a 0%, #0f172a 50%, #0a1628 100%)',
      },
      animation: {
        'pulse-slow': 'pulse 2s infinite',
        'slide-in': 'slideIn 0.2s ease-out',
        'fade-in': 'fadeIn 0.2s ease-out',
      },
      keyframes: {
        slideIn: {
          '0%': { transform: 'translateX(100%)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
