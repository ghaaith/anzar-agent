"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";
import { login, register } from "@/lib/api";

export default function LoginPage() {
  const [isSignIn, setIsSignIn] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const { setAuth } = useAuth();
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      if (isSignIn) {
        const data = await login(email, password);
        setAuth(data.access_token, data.refresh_token, data.user);
      } else {
        const data = await register(name, email, password);
        setAuth(data.access_token, data.refresh_token, data.user);
      }
      router.push("/chat/");
    } catch (err: any) {
      setError(err.message || "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen w-full flex overflow-hidden">
      {/* Left Panel — Brand */}
      <section className="relative w-[40%] bg-surface-lowest items-center justify-center overflow-hidden border-r border-white/5 hidden md:flex">
        {/* Subtle grid pattern */}
        <div
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage:
              "linear-gradient(rgba(173,198,255,0.3) 1px, transparent 1px), linear-gradient(90deg, rgba(173,198,255,0.3) 1px, transparent 1px)",
            backgroundSize: "48px 48px",
          }}
        />

        {/* Ambient glows */}
        <div className="absolute top-1/3 left-1/4 w-[400px] h-[400px] bg-primary/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-1/4 right-1/4 w-[300px] h-[300px] bg-tertiary/5 rounded-full blur-[100px]" />

        {/* Floating dots */}
        <div className="absolute inset-0 overflow-hidden">
          {[...Array(12)].map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 bg-primary/20 rounded-full animate-pulse-slow"
              style={{
                left: `${10 + (i * 7.5) % 85}%`,
                top: `${15 + (i * 13) % 70}%`,
                animationDelay: `${i * 0.4}s`,
              }}
            />
          ))}
        </div>

        <div className="relative z-10 flex flex-col items-center text-center px-12">
          <div className="mb-8 relative">
            <div className="absolute -inset-4 bg-primary/5 rounded-full blur-xl" />
            <div className="relative w-28 h-28 rounded-2xl bg-surface-container/80 border border-white/[0.06] flex items-center justify-center shadow-glow-lg">
              <span
                className="material-symbols-outlined text-primary text-6xl"
                style={{ fontVariationSettings: "'FILL' 1" }}
              >
                cloudy_snowing
              </span>
            </div>
          </div>

          <h1 className="font-display text-6xl font-bold text-white tracking-tight mb-3">
            Anzar
          </h1>
          <p className="text-on-surface-variant text-lg max-w-xs leading-relaxed">
            Your AI-Powered Code Companion
          </p>

          <div className="mt-16 flex gap-3">
            <span className="font-mono text-[10px] text-primary/70 uppercase tracking-[0.15em] border border-primary/10 px-3 py-1.5 rounded-md bg-primary/[0.03]">
              v0.1.0
            </span>
            <span className="font-mono text-[10px] text-tertiary/70 uppercase tracking-[0.15em] border border-tertiary/10 px-3 py-1.5 rounded-md bg-tertiary/[0.03]">
              Neural Core
            </span>
          </div>
        </div>

        {/* Corner accents */}
        <div className="absolute top-8 left-8 w-12 h-12 border-t border-l border-white/[0.04] rounded-tl-lg" />
        <div className="absolute bottom-8 right-8 w-12 h-12 border-b border-r border-white/[0.04] rounded-br-lg" />
      </section>

      {/* Right Panel — Auth Form */}
      <main className="relative flex-1 flex items-center justify-center bg-background p-8 md:p-12">
        {/* Background glows */}
        <div className="absolute top-1/4 -right-16 w-96 h-96 bg-primary/[0.04] blur-[120px] rounded-full" />
        <div className="absolute bottom-1/4 -left-16 w-80 h-80 bg-secondary/[0.03] blur-[100px] rounded-full" />

        <div className="glass-card w-full max-w-md p-8 rounded-2xl relative animate-fade-in">
          {/* Tab Switcher */}
          <div className="flex bg-surface-lowest/60 rounded-xl p-1 mb-8">
            <button
              onClick={() => { setIsSignIn(true); setError(""); }}
              className={`flex-1 py-2.5 rounded-lg text-sm font-semibold transition-all duration-200 ${isSignIn
                ? "bg-primary/15 text-primary shadow-sm"
                : "text-on-surface-variant hover:text-on-surface"
                }`}
            >
              Sign In
            </button>
            <button
              onClick={() => { setIsSignIn(false); setError(""); }}
              className={`flex-1 py-2.5 rounded-lg text-sm font-semibold transition-all duration-200 ${!isSignIn
                ? "bg-primary/15 text-primary shadow-sm"
                : "text-on-surface-variant hover:text-on-surface"
                }`}
            >
              Create Account
            </button>
          </div>

          {/* Heading */}
          <div className="mb-6">
            <h2 className="text-headline-md text-white mb-1">
              {isSignIn ? "Welcome back" : "Create your account"}
            </h2>
            <p className="text-on-surface-variant text-body-md">
              {isSignIn
                ? "Sign in to continue building."
                : "Start building with your AI code companion."}
            </p>
          </div>

          {/* Error */}
          {error && (
            <div className="mb-5 p-3 rounded-xl bg-error/10 border border-error/20 text-error text-sm animate-slide-up">
              {error}
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {!isSignIn && (
              <div className="space-y-1.5">
                <label className="block font-mono text-label-sm text-on-surface-variant">
                  FULL NAME
                </label>
                <div className="relative">
                  <span className="material-symbols-outlined absolute left-3.5 top-1/2 -translate-y-1/2 text-on-surface-variant/60 text-lg">

                  </span>
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="John Doe"
                    className="input-field w-full pl-11"
                    required={!isSignIn}
                  />
                </div>
              </div>
            )}

            <div className="space-y-1.5">
              <label className="block font-mono text-label-sm text-on-surface-variant">
                EMAIL ADDRESS
              </label>
              <div className="relative">
                <span className="material-symbols-outlined absolute left-3.5 top-1/2 -translate-y-1/2 text-on-surface-variant/60 text-lg">
                </span>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="dev@anzar.ai"
                  className="input-field w-full pl-11"
                  required
                />
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="flex justify-between items-center">
                <label className="block font-mono text-label-sm text-on-surface-variant">
                  PASSWORD
                </label>
                {isSignIn && (
                  <button type="button" className="text-primary/80 text-label-sm hover:text-primary transition-colors">
                    Forgot?
                  </button>
                )}
              </div>
              <div className="relative">
                <span className="material-symbols-outlined absolute left-3.5 top-1/2 -translate-y-1/2 text-on-surface-variant/60 text-lg">
                  *                </span>
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter password"
                  className="input-field w-full pl-11 pr-11"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-on-surface-variant/60 hover:text-on-surface transition-colors"
                >
                  <span className="material-symbols-outlined text-lg">
                    {showPassword ? "visibility_off" : "visibility"}
                  </span>
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full mt-2 flex items-center justify-center gap-2"
            >
              {loading ? (
                <div className="w-5 h-5 border-2 border-on-primary border-t-transparent rounded-full animate-spin" />
              ) : (
                <>
                  <span>{isSignIn ? "Sign In" : "Create Account"}</span>
                  <span className="material-symbols-outlined text-[18px]">

                  </span>
                </>
              )}
            </button>
          </form>

          {/* Divider */}
          <div className="relative my-8">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-white/[0.05]" />
            </div>
            <div className="relative flex justify-center">
              <span className="bg-surface-lowest/80 px-3 text-on-surface-variant/60 font-mono text-[11px] uppercase tracking-[0.1em]">
                or continue with
              </span>
            </div>
          </div>

          {/* OAuth Buttons */}
          <div className="grid grid-cols-2 gap-3">
            <button className="flex items-center justify-center gap-2.5 py-2.5 px-4 rounded-xl bg-surface-lowest/60 border border-white/[0.05] hover:bg-surface-low hover:border-white/[0.08] transition-all duration-200 text-sm font-medium text-on-surface-variant hover:text-on-surface">
              <svg className="w-4.5 h-4.5" viewBox="0 0 24 24" width="18" height="18">
                <path d="M12.48 10.92v3.28h7.84c-.24 1.84-.908 3.152-1.896 4.144-1.232 1.232-3.152 2.576-6.424 2.576-5.112 0-9.152-4.144-9.152-9.256s4.04-9.256 9.152-9.256c2.776 0 4.832 1.096 6.328 2.512l2.304-2.304C18.576 1.016 15.864 0 12.48 0 5.68 0 0 5.68 0 12.48s5.68 12.48 12.48 12.48c3.68 0 6.464-1.216 8.704-3.552 2.312-2.312 3.04-5.552 3.04-8.2s-.08-4.32-.24-5.808h-11.48z" fill="#EA4335" />
              </svg>
              Google
            </button>
            <button className="flex items-center justify-center gap-2.5 py-2.5 px-4 rounded-xl bg-surface-lowest/60 border border-white/[0.05] hover:bg-surface-low hover:border-white/[0.08] transition-all duration-200 text-sm font-medium text-on-surface-variant hover:text-on-surface">
              <svg className="w-4.5 h-4.5" viewBox="0 0 24 24" width="18" height="18" fill="#c2c6d6">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
              </svg>
              GitHub
            </button>
          </div>

          {/* Footer */}
          <div className="mt-8 pt-5 border-t border-white/[0.04] flex justify-between items-center">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
              <span className="font-mono text-[10px] text-green-500/70 uppercase tracking-[0.1em]">
                Systems online
              </span>
            </div>
            <div className="flex gap-4">
              <span className="font-mono text-[10px] text-on-surface-variant/50 uppercase tracking-[0.1em] cursor-pointer hover:text-on-surface-variant transition-colors">
                Privacy
              </span>
              <span className="font-mono text-[10px] text-on-surface-variant/50 uppercase tracking-[0.1em] cursor-pointer hover:text-on-surface-variant transition-colors">
                Terms
              </span>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
