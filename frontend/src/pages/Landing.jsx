import React from 'react'
import { Link } from 'react-router-dom'
import { useTheme } from '../ThemeContext.jsx'

const features = [
  {
    icon: '⚡',
    title: 'Faster Review',
    text: 'Get quick AI-assisted insights in seconds.'
  },
  {
    icon: '🎯',
    title: 'AI Classification',
    text: 'AI-assisted tumor detection and classification.'
  },
  {
    icon: '👁️',
    title: 'Visual Support',
    text: 'Clear visual explanations to support review.'
  }
]

const steps = [
  {
    number: 1,
    icon: '☁️',
    title: 'Upload MRI',
    text: 'Upload your brain MRI image securely.'
  },
  {
    number: 2,
    icon: '🤖',
    title: 'AI Analysis',
    text: 'AI analyzes the image and detects patterns.'
  },
  {
    number: 3,
    icon: '📋',
    title: 'Review Result',
    text: 'Get classification, confidence and visual results.'
  },
  {
    number: 4,
    icon: '👨‍⚕️',
    title: 'Doctor Review',
    text: 'The result can be reviewed by a medical professional.'
  }
]

export default function Landing() {
  const { darkMode, toggleTheme } = useTheme();
  return (
    <div className="min-h-screen bg-[#f7f9ff] dark:bg-slate-950 text-[#10194f] dark:text-white">

      {/* ================= NAVBAR ================= */}
      <header className="px-4 sm:px-6 lg:px-9 pt-3">
        <nav className="max-w-[1460px] mx-auto bg-white dark:bg-slate-900 rounded-[28px] shadow-[0_10px_35px_rgba(40,70,150,0.10)] px-7 py-3">

          <div className="flex items-center justify-between">

            {/* LOGO */}
            <Link to="/" className="flex items-center gap-3">

              <div className="w-14 h-14 flex items-center justify-center">
                <span className="text-[43px]">🧠</span>
              </div>

              <div>
                <div className="text-[27px] font-extrabold tracking-tight">
                  Neuro<span className="text-blue-600">Scan</span>
                </div>

                <div className="text-[12px] font-medium text-slate-500 dark:text-slate-400 -mt-1">
                  AI-Powered Brain MRI Analysis
                </div>
              </div>

            </Link>

            {/* NAVIGATION */}
            <div className="hidden lg:flex items-center gap-12 text-[17px] font-semibold">

              <a
                href="#home"
                className="text-blue-600 relative py-5"
              >
                Home
                <span className="absolute left-0 right-0 -bottom-1 h-[3px] bg-blue-600 rounded-full" />
              </a>

              <a
                href="#how-it-works"
                className="hover:text-blue-600 transition"
              >
                How It Works
              </a>

              <a
                href="#about"
                className="hover:text-blue-600 transition"
              >
                About
              </a>

              <a
                href="#features"
                className="hover:text-blue-600 transition"
              >
                Features
              </a>

            </div>

            {/* RIGHT SIDE */}
            <div className="flex items-center gap-3">

              <button
  type="button"
  onClick={toggleTheme}
  className="
    flex items-center gap-2
    px-5 py-2.5
    rounded-full
    border border-slate-200 dark:border-slate-700
    bg-white dark:bg-slate-900
    text-slate-700 dark:text-slate-200
    shadow-sm
    hover:shadow-md
    transition
  "
>
  <span className="text-lg">
    {darkMode ? "☀️" : "🌙"}
  </span>

  <span className="font-medium">
    {darkMode ? "Light" : "Dark"}
  </span>
</button>

              <Link
                to="/login"
                className="hidden sm:flex items-center gap-2 px-6 py-3 rounded-2xl border-2 border-blue-600 text-blue-600 font-semibold hover:bg-blue-50 dark:hover:bg-blue-950 transition"
              >
                <span>♙</span>
                Login
              </Link>

              <Link
                to="/signup"
                className="hidden sm:block px-7 py-3 rounded-2xl bg-gradient-to-r from-indigo-600 to-blue-600 text-white font-bold shadow-lg hover:shadow-blue-500/30 transition"
              >
                Get Started
              </Link>

            </div>

          </div>
        </nav>
      </header>

      {/* ================= HERO ================= */}
      <section
        id="home"
        className="max-w-[1380px] mx-auto px-6 lg:px-10"
      >

        <div className="grid lg:grid-cols-[0.9fr_1.1fr] items-center min-h-[470px]">

          {/* LEFT CONTENT */}
          <div className="pt-12 lg:pt-4">

            <h1 className="text-[50px] sm:text-[58px] lg:text-[55px] xl:text-[62px] leading-[1.04] font-black tracking-tight">

              <span className="block text-[#11194d] dark:text-white">
                AI-ASSISTED
              </span>

              <span className="block bg-gradient-to-r from-purple-600 via-blue-600 to-indigo-600 bg-clip-text text-transparent">
                BRAIN MRI ANALYSIS
              </span>

            </h1>

            {/* SUBTITLE */}
            <div className="mt-7">

              <h2 className="text-[22px] sm:text-[23px] font-bold">
                Faster MRI Analysis. Smarter Clinical Support.
              </h2>

              <div className="flex items-center gap-2 mt-3">
                <div className="w-10 h-[3px] bg-blue-600 rounded-full" />
                <div className="w-2 h-[3px] bg-blue-300 rounded-full" />
              </div>

            </div>

            <p className="mt-4 text-[17px] leading-7 text-slate-600 dark:text-slate-300 max-w-[550px]">
              Upload a brain MRI and receive an AI-assisted
              classification with confidence and visual explanation.
            </p>

            {/* BUTTONS */}
            <div className="flex flex-wrap gap-5 mt-7">

              <Link
                to="/analyze"
                className="flex items-center gap-3 px-7 py-4 rounded-xl bg-gradient-to-r from-purple-600 to-blue-600 text-white font-bold text-[16px] shadow-xl shadow-blue-500/20 hover:-translate-y-0.5 transition"
              >
                <span className="text-2xl">☁</span>
                Start Analysis
              </Link>

              <a
                href="#how-it-works"
                className="flex items-center gap-3 px-7 py-4 rounded-xl border-2 border-blue-600 text-blue-700 dark:text-blue-300 font-bold text-[16px] hover:bg-blue-50 dark:hover:bg-blue-950 transition"
              >
                <span className="text-xl">▷</span>
                Explore How It Works
              </a>

            </div>

          </div>

          {/* ================= BRAIN IMAGE ================= */}
          <div className="relative flex justify-center items-center min-h-[430px]">

            {/* Glow behind brain */}
            <div className="absolute w-[430px] h-[430px] bg-blue-400/20 dark:bg-blue-600/10 blur-3xl rounded-full" />

            <img
              src="/neuroscan-brain-hero.png"
              alt="AI assisted brain MRI visualization"
              className="relative z-10 w-full max-w-[620px] object-contain drop-shadow-[0_20px_45px_rgba(37,99,235,0.22)]"
            />

          </div>

        </div>

      </section>

      {/* ================= FEATURES ================= */}
      <section
        id="features"
        className="max-w-[1380px] mx-auto px-6 lg:px-10"
      >

        <div className="bg-white dark:bg-slate-900 rounded-[25px] shadow-[0_8px_30px_rgba(30,60,130,0.08)] px-6 sm:px-8 py-6">

          <div className="grid md:grid-cols-3">

            {features.map((feature, index) => (

              <div
                key={feature.title}
                className={`flex items-center gap-5 px-6 py-3 ${
                  index !== 0
                    ? 'border-l border-slate-200 dark:border-slate-700'
                    : ''
                }`}
              >

                <div className="w-[76px] h-[76px] shrink-0 rounded-2xl bg-blue-50 dark:bg-blue-950 flex items-center justify-center text-[34px]">
                  {feature.icon}
                </div>

                <div>

                  <h3 className="font-extrabold text-[18px]">
                    {feature.title}
                  </h3>

                  <p className="mt-1 text-[14px] leading-6 text-slate-600 dark:text-slate-300">
                    {feature.text}
                  </p>

                </div>

              </div>

            ))}

          </div>

        </div>

      </section>

      {/* ================= HOW IT HELPS ================= */}
      <section
        id="how-it-works"
        className="max-w-[1380px] mx-auto px-6 lg:px-10 pt-7 pb-8"
      >

        <div className="text-center">

          <h2 className="text-[28px] font-black">
            HOW IT HELPS
          </h2>

          <div className="w-10 h-[3px] bg-blue-600 rounded-full mx-auto mt-3" />

        </div>

        <div className="grid md:grid-cols-4 mt-7">

          {steps.map((step, index) => (

            <div
              key={step.title}
              className="relative text-center"
            >

              {/* CONNECTING LINE */}
              {index < steps.length - 1 && (
                <div className="hidden md:block absolute top-[46px] left-[58%] right-[-42%] border-t-2 border-dashed border-blue-200 dark:border-blue-900" />
              )}

              {/* STEP CIRCLE */}
              <div className="relative mx-auto w-[92px] h-[92px] rounded-full bg-white dark:bg-slate-900 border border-blue-100 dark:border-blue-900 shadow-lg flex items-center justify-center text-[36px]">

                {step.icon}

                <span className="absolute -top-2 -left-2 w-8 h-8 rounded-full bg-blue-600 text-white text-sm font-bold flex items-center justify-center">
                  {step.number}
                </span>

              </div>

              <h3 className="mt-3 font-extrabold text-[18px]">
                {step.title}
              </h3>

              <p className="mt-1 text-[14px] leading-6 text-slate-600 dark:text-slate-300 max-w-[230px] mx-auto">
                {step.text}
              </p>

            </div>

          ))}

        </div>

      </section>

      {/* ================= ABOUT ================= */}
      <section
        id="about"
        className="max-w-[1380px] mx-auto px-6 lg:px-10 pb-7"
      >

        <div className="bg-blue-50/70 dark:bg-blue-950/30 border border-blue-100 dark:border-blue-900 rounded-2xl px-6 py-5">

          <p className="text-center text-sm text-slate-600 dark:text-slate-300">

            <span className="font-bold text-blue-600">
              About NeuroScan:
            </span>{' '}

            An AI-assisted platform designed to reduce the time required
            for initial brain MRI image review and provide useful
            information for professional evaluation.

          </p>

        </div>

      </section>

      {/* ================= DISCLAIMER ================= */}
      <section
        className="max-w-[1380px] mx-auto px-6 lg:px-10 pb-8"
      >

        <div className="bg-blue-50 dark:bg-blue-950/40 rounded-2xl border border-blue-100 dark:border-blue-900 px-7 py-4">

          <div className="flex items-center gap-4">

            <div className="w-10 h-10 rounded-xl bg-blue-600 text-white flex items-center justify-center text-xl shrink-0">
              ✓
            </div>

            <p className="text-[14px] text-slate-700 dark:text-slate-300">

              <span className="font-bold text-blue-600">
                Important:
              </span>{' '}

              AI output is decision-support and does not replace
              professional medical diagnosis.

            </p>

          </div>

        </div>

      </section>

      {/* ================= FOOTER ================= */}
      <footer className="border-t border-slate-200 dark:border-slate-800 py-5">

        <div className="max-w-[1380px] mx-auto px-6 flex justify-between items-center">

          <div className="font-bold">
            Neuro<span className="text-blue-600">Scan</span>
          </div>

          <div className="text-xs text-slate-400">
            AI-assisted brain MRI analysis
          </div>

        </div>

      </footer>

    </div>
  )
}