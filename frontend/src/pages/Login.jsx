import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth.jsx";

export default function Login() {
  const navigate = useNavigate();
  const { login } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();

    setError("");
    setLoading(true);

    try {
      await login(email, password);

      navigate("/dashboard");
    } catch (err) {
      console.error("Login error:", err);

      const message =
        err?.response?.data?.detail ||
        err?.response?.data?.message ||
        "Login failed. Please check your email and password.";

      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f7f9ff] dark:bg-slate-950 flex items-center justify-center px-4">

      {/* HOME BUTTON */}
      <div className="absolute top-5 left-5">
        <Link
          to="/"
          className="text-blue-600 font-semibold hover:underline"
        >
          ← Go to Home
        </Link>
      </div>

      {/* LOGIN CARD */}
      <div className="w-full max-w-md">

        <div className="bg-white dark:bg-slate-900 rounded-3xl shadow-xl border border-slate-200 dark:border-slate-800 p-8">

          {/* LOGO */}
          <div className="text-center mb-8">

            <div className="text-5xl mb-3">
              🧠
            </div>

            <h1 className="text-3xl font-extrabold text-slate-900 dark:text-white">
              Neuro<span className="text-blue-600">Scan</span>
            </h1>

            <p className="mt-2 text-slate-500 dark:text-slate-400">
              AI-Powered Brain MRI Analysis
            </p>

          </div>

          {/* TITLE */}
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-6">
            Login
          </h2>

          {/* ERROR MESSAGE */}
          {error && (
            <div className="mb-5 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
              {error}
            </div>
          )}

          {/* FORM */}
          <form onSubmit={handleLogin} className="space-y-5">

            {/* EMAIL */}
            <div>
              <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
                Email
              </label>

              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Enter your email"
                required
                autoComplete="email"
                className="
                  w-full px-4 py-3 rounded-xl
                  border border-slate-300 dark:border-slate-700
                  bg-white dark:bg-slate-950
                  text-slate-900 dark:text-white
                  placeholder-slate-400
                  focus:outline-none
                  focus:ring-2
                  focus:ring-blue-500
                "
              />
            </div>

            {/* PASSWORD */}
            <div>
              <label className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
                Password
              </label>

              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                required
                autoComplete="current-password"
                className="
                  w-full px-4 py-3 rounded-xl
                  border border-slate-300 dark:border-slate-700
                  bg-white dark:bg-slate-950
                  text-slate-900 dark:text-white
                  placeholder-slate-400
                  focus:outline-none
                  focus:ring-2
                  focus:ring-blue-500
                "
              />
            </div>

            {/* LOGIN BUTTON */}
            <button
              type="submit"
              disabled={loading}
              className="
                w-full py-3.5 rounded-xl
                bg-gradient-to-r from-indigo-600 to-blue-600
                text-white font-bold
                hover:shadow-lg
                transition
                disabled:opacity-60
                disabled:cursor-not-allowed
              "
            >
              {loading ? "Logging in..." : "Login"}
            </button>

          </form>

          {/* SIGNUP */}
          <p className="text-center mt-6 text-slate-600 dark:text-slate-400">
            Don't have an account?{" "}

            <Link
              to="/signup"
              className="text-blue-600 font-semibold hover:underline"
            >
              Create Account
            </Link>
          </p>

        </div>

      </div>

    </div>
  );
}