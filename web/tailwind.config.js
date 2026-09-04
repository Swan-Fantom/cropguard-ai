/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // CropGuard leaf-green accent.
        brand: {
          50: "#f0f9f1",
          100: "#dcf0de",
          500: "#2f9e44",
          600: "#2b8a3e",
          700: "#237032",
          800: "#1d5a2a",
        },
      },
    },
  },
  plugins: [],
};
