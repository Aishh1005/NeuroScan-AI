import React from "react";
import { useTheme } from "../ThemeContext.jsx";

export function ThemeToggle() {
  const { darkMode, toggleTheme } = useTheme();

  return (
    <button
      type="button"
      onClick={toggleTheme}
      className="flex items-center gap-2 px-4 py-2 rounded-xl border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
    >
      <span className="text-lg">
        {darkMode ? "☀️" : "🌙"}
      </span>

      <span className="font-medium">
        {darkMode ? "Light" : "Dark"}
      </span>
    </button>
  );
}