"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";
import { getSettings, updateSettings } from "@/lib/api";

interface ProviderConfig {
  id: string;
  name: string;
  description: string;
  icon: string;
  iconColor: string;
  accentColor: string;
  models: string[];
  needsKey: boolean;
}

const PROVIDERS: ProviderConfig[] = [
  {
    id: "openrouter",
    name: "OpenRouter",
    description: "Access 200+ open models via a single API. Free tier available.",
    icon: "router",
    iconColor: "text-primary",
    accentColor: "from-primary/10 to-primary/5 border-primary/10",
    models: [
      "openai/gpt-oss-20b:free",
      "poolside/laguna-xs-2.1:free",
      "nvidia/nemotron-3-nano-30b-a3b:free",
      "nvidia/nemotron-3-super-120b-a12b:free",
    ],
    needsKey: true,
  },
  {
    id: "gemini",
    name: "Google Gemini",
    description: "Google's multimodal AI models with long context.",
    icon: "auto_awesome",
    iconColor: "text-blue-400",
    accentColor: "from-blue-500/10 to-blue-500/5 border-blue-500/10",
    models: ["gemini-2.0-flash", "gemini-2.5-pro"],
    needsKey: true,
  },
  {
    id: "openai",
    name: "OpenAI",
    description: "GPT-4o, GPT-4 Turbo, and more.",
    icon: "bolt",
    iconColor: "text-emerald-400",
    accentColor: "from-emerald-500/10 to-emerald-500/5 border-emerald-500/10",
    models: ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
    needsKey: true,
  },
  {
    id: "anthropic",
    name: "Anthropic",
    description: "Claude models — excellent for coding.",
    icon: "hive",
    iconColor: "text-amber-300",
    accentColor: "from-amber-500/10 to-amber-500/5 border-amber-500/10",
    models: ["claude-sonnet-4-20250514", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
    needsKey: true,
  },
  {
    id: "groq",
    name: "Groq",
    description: "Ultra-fast inference on open models.",
    icon: "speed",
    iconColor: "text-orange-400",
    accentColor: "from-orange-500/10 to-orange-500/5 border-orange-500/10",
    models: ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
    needsKey: true,
  },
  {
    id: "ollama",
    name: "Ollama",
    description: "Run models locally on your machine.",
    icon: "local_fire_department",
    iconColor: "text-rose-400",
    accentColor: "from-rose-500/10 to-rose-500/5 border-rose-500/10",
    models: ["llama3.1", "codellama", "deepseek-coder"],
    needsKey: false,
  },
];

export default function SettingsPage() {
  const { user, token, loading: authLoading, logout } = useAuth();
  const router = useRouter();

  const [settings, setSettings] = useState({
    provider: "groq",
    model: "",
    apiKey: "",
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [showKey, setShowKey] = useState(false);

  useEffect(() => {
    if (!authLoading && !user) router.replace("/login/");
  }, [user, authLoading, router]);

  useEffect(() => {
    if (token) {
      getSettings()
        .then((s) => {
          setSettings({
            provider: s.provider || "openrouter",
            model: s.model || "",
            apiKey: "",
          });
        })
        .catch(console.error)
        .finally(() => setLoading(false));
    }
  }, [token]);

  const handleSave = async () => {
    setSaving(true);
    setSaved(false);
    try {
      await updateSettings({
        provider: settings.provider,
        model: settings.model || undefined,
        api_key: settings.apiKey || undefined,
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      console.error("Failed to save settings:", err);
    } finally {
      setSaving(false);
    }
  };

  const currentProvider = PROVIDERS.find((p) => p.id === settings.provider);

  if (authLoading || loading) {
    return (
      <div className="h-screen w-full flex items-center justify-center bg-background">
        <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="h-screen w-full flex overflow-hidden">
      {/* Sidebar */}
      <aside className="flex flex-col h-full w-[260px] shrink-0 bg-surface-lowest/60 border-r border-white/[0.05]">
        {/* Brand */}
        <div className="px-5 py-6 flex items-center gap-2.5">
          <img src="/assets/logo.png" alt="Anzar" className="w-8 h-8 rounded-lg object-contain" />
          <span className="font-display text-xl font-bold text-primary tracking-tight">Anzar</span>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-2 space-y-1">
          <button
            onClick={() => router.push("/chat/")}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-on-surface-variant hover:text-on-surface hover:bg-white/[0.03] transition-all duration-150"
          >
            <span className="material-symbols-outlined text-xl">chat</span>
            <span className="text-body-md">Chat</span>
          </button>
          <div className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl bg-primary/10 text-primary">
            <span className="material-symbols-outlined text-xl">memory</span>
            <span className="text-body-md font-medium">LLM Providers</span>
          </div>
          <button
            onClick={() => router.push("/profile/")}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-on-surface-variant hover:text-on-surface hover:bg-white/[0.03] transition-all duration-150"
          >
            <span className="material-symbols-outlined text-xl">person</span>
            <span className="text-body-md">Profile</span>
          </button>
          <button
            onClick={() => router.push("/usage/")}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-on-surface-variant hover:text-on-surface hover:bg-white/[0.03] transition-all duration-150"
          >
            <span className="material-symbols-outlined text-xl">monitoring</span>
            <span className="text-body-md">Usage</span>
          </button>
        </nav>

        {/* User */}
        <div className="border-t border-white/[0.04] p-3">
          <div className="flex items-center gap-2.5 px-2 py-2">
            <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-xs font-bold text-primary border border-primary/10">
              {user?.name?.charAt(0)?.toUpperCase() || "U"}
            </div>
            <div className="min-w-0">
              <div className="text-sm font-medium text-on-surface truncate">{user?.name || "User"}</div>
              <div className="text-[10px] text-on-surface-variant/50 font-mono">Free Plan</div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 flex flex-col bg-background relative overflow-hidden">
        {/* Ambient */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute -top-32 -right-32 w-[500px] h-[500px] bg-primary/[0.03] rounded-full blur-[150px]" />
        </div>

        {/* Header */}
        <header className="h-14 flex justify-between items-center px-6 border-b border-white/[0.05] bg-surface-lowest/60 backdrop-blur-xl z-40 shrink-0">
          <span className="font-mono text-[11px] text-on-surface-variant/50 uppercase tracking-[0.1em]">
            Configuration
          </span>
          <button
            onClick={() => { logout(); router.replace("/login/"); }}
            className="btn-ghost p-2 rounded-lg hover:text-error"
          >
            <span className="material-symbols-outlined text-xl">logout</span>
          </button>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-y-auto scroll-hide relative z-10">
          <div className="max-w-3xl mx-auto px-6 py-8 space-y-8">
            {/* Title */}
            <div>
              <h1 className="text-headline-lg text-white mb-1">LLM Providers</h1>
              <p className="text-on-surface-variant/60 text-body-md">
                Configure your AI model providers and API keys
              </p>
            </div>

            {/* Provider Cards */}
            <div className="space-y-3">
              {PROVIDERS.map((provider) => {
                const isActive = settings.provider === provider.id;
                return (
                  <div
                    key={provider.id}
                    className={`rounded-2xl border transition-all duration-200 overflow-hidden ${
                      isActive
                        ? "bg-surface-container/60 border-white/[0.08] shadow-glow"
                        : "bg-surface-lowest/30 border-white/[0.04] hover:border-white/[0.06]"
                    }`}
                  >
                    {/* Provider Header */}
                    <div
                      className="flex items-center justify-between p-5 cursor-pointer"
                      onClick={() => {
                        if (!isActive) {
                          setSettings((s) => ({
                            ...s,
                            provider: provider.id,
                            model: provider.models[0],
                          }));
                        }
                      }}
                    >
                      <div className="flex items-center gap-4">
                        <div
                          className={`w-11 h-11 rounded-xl flex items-center justify-center bg-gradient-to-br ${provider.accentColor}`}
                        >
                          <span className={`material-symbols-outlined text-xl ${provider.iconColor}`}>
                            {provider.icon}
                          </span>
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <h3 className="text-body-lg font-semibold text-white">{provider.name}</h3>
                            {isActive && (
                              <span className="text-[9px] font-mono uppercase tracking-[0.15em] text-primary bg-primary/10 px-2 py-0.5 rounded-full">
                                Active
                              </span>
                            )}
                          </div>
                          <p className="text-body-sm text-on-surface-variant/50 mt-0.5">{provider.description}</p>
                        </div>
                      </div>

                      {/* Toggle */}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSettings((s) => ({
                            ...s,
                            provider: provider.id,
                            model: provider.models[0],
                          }));
                        }}
                        className={`relative w-10 h-[22px] rounded-full transition-colors duration-200 ${
                          isActive ? "bg-primary" : "bg-surface-container-highest"
                        }`}
                      >
                        <div
                          className={`absolute top-[3px] w-4 h-4 rounded-full transition-all duration-200 ${
                            isActive
                              ? "left-[22px] bg-on-primary"
                              : "left-[3px] bg-on-surface-variant/60"
                          }`}
                        />
                      </button>
                    </div>

                    {/* Expanded Config */}
                    {isActive && (
                      <div className="px-5 pb-5 space-y-4 animate-slide-up border-t border-white/[0.04]">
                        {/* API Key */}
                        {provider.needsKey && (
                          <div className="pt-4 space-y-1.5">
                            <label className="block font-mono text-label-sm text-on-surface-variant/60">
                              API KEY
                            </label>
                            <div className="relative">
                              <input
                                type={showKey ? "text" : "password"}
                                value={settings.apiKey}
                                onChange={(e) => setSettings((s) => ({ ...s, apiKey: e.target.value }))}
                                placeholder={`Enter your ${provider.name} API key...`}
                                className="input-field w-full pr-11 text-sm"
                              />
                              <button
                                type="button"
                                onClick={() => setShowKey(!showKey)}
                                className="absolute right-3 top-1/2 -translate-y-1/2 text-on-surface-variant/40 hover:text-on-surface-variant transition-colors"
                              >
                                <span className="material-symbols-outlined text-lg">
                                  {showKey ? "visibility_off" : "visibility"}
                                </span>
                              </button>
                            </div>
                          </div>
                        )}

                        {/* Model */}
                        <div className="space-y-1.5">
                          <label className="block font-mono text-label-sm text-on-surface-variant/60">
                            MODEL
                          </label>
                          <select
                            value={settings.model}
                            onChange={(e) => setSettings((s) => ({ ...s, model: e.target.value }))}
                            className="input-field w-full text-sm appearance-none"
                          >
                            {provider.models.map((m) => (
                              <option key={m} value={m}>{m}</option>
                            ))}
                          </select>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="shrink-0 border-t border-white/[0.05] bg-surface-lowest/60 backdrop-blur-xl px-6 py-4 flex items-center justify-between relative z-40">
          <button
            onClick={() => router.push("/chat/")}
            className="btn-ghost flex items-center gap-2 text-sm"
          >
            <span className="material-symbols-outlined text-lg">arrow_back</span>
            Back to Chat
          </button>

          <div className="flex items-center gap-4">
            {saved && (
              <span className="text-green-400 font-mono text-xs uppercase tracking-[0.1em] animate-fade-in">
                Saved
              </span>
            )}
            <button
              onClick={handleSave}
              disabled={saving}
              className="btn-primary flex items-center gap-2"
            >
              {saving ? (
                <div className="w-4 h-4 border-2 border-on-primary border-t-transparent rounded-full animate-spin" />
              ) : (
                <span className="material-symbols-outlined text-lg">check</span>
              )}
              {saving ? "Saving..." : "Save Changes"}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
