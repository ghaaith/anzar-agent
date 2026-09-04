"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";
import { getUsageSummary, getUsageHistory } from "@/lib/api";

interface DailyUsage {
  date: string;
  tokens: number;
  actions: number;
}

interface UsageRecord {
  id: string;
  action: string;
  tokens_used: number;
  created_at: string;
}

export default function UsagePage() {
  const { user, loading: authLoading, logout } = useAuth();
  const router = useRouter();

  const [summary, setSummary] = useState({
    total_tokens: 0,
    total_actions: 0,
    chat_count: 0,
    daily: [] as DailyUsage[],
    plan: "free",
    token_limit: 10000,
    token_usage_pct: 0,
  });
  const [history, setHistory] = useState<UsageRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!authLoading && !user) router.replace("/login/");
  }, [user, authLoading, router]);

  useEffect(() => {
    if (user) {
      Promise.all([getUsageSummary(), getUsageHistory()])
        .then(([s, h]) => { setSummary(s); setHistory(h); })
        .catch(console.error)
        .finally(() => setLoading(false));
    }
  }, [user]);

  const formatTokens = (n: number) => {
    if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
    if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
    return n.toString();
  };

  const formatTime = (dateStr: string) => {
    const d = new Date(dateStr);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  const maxDailyTokens = Math.max(...summary.daily.map((d) => d.tokens), 1);

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
        <div className="px-5 py-6 flex items-center gap-2.5">
          <img src="/assets/logo.png" alt="Anzar" className="w-8 h-8 rounded-lg object-contain" />
          <span className="font-display text-xl font-bold text-primary tracking-tight">Anzar</span>
        </div>

        <nav className="flex-1 px-3 py-2 space-y-1">
          <button
            onClick={() => router.push("/chat/")}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-on-surface-variant hover:text-on-surface hover:bg-white/[0.03] transition-all duration-150"
          >
            <span className="material-symbols-outlined text-xl">chat</span>
            <span className="text-body-md">Chat</span>
          </button>
          <button
            onClick={() => router.push("/settings/")}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-on-surface-variant hover:text-on-surface hover:bg-white/[0.03] transition-all duration-150"
          >
            <span className="material-symbols-outlined text-xl">memory</span>
            <span className="text-body-md">LLM Providers</span>
          </button>
          <button
            onClick={() => router.push("/profile/")}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-on-surface-variant hover:text-on-surface hover:bg-white/[0.03] transition-all duration-150"
          >
            <span className="material-symbols-outlined text-xl">person</span>
            <span className="text-body-md">Profile</span>
          </button>
          <div className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl bg-primary/10 text-primary">
            <span className="material-symbols-outlined text-xl">monitoring</span>
            <span className="text-body-md font-medium">Usage</span>
          </div>
        </nav>

        <div className="border-t border-white/[0.04] p-3">
          <div className="flex items-center gap-2.5 px-2 py-2">
            <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-xs font-bold text-primary border border-primary/10">
              {user?.name?.charAt(0)?.toUpperCase() || "U"}
            </div>
            <div className="min-w-0">
              <div className="text-sm font-medium text-on-surface truncate">{user?.name || "User"}</div>
              <div className="text-[10px] text-on-surface-variant/50 font-mono capitalize">{summary.plan} Plan</div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 flex flex-col bg-background relative overflow-hidden">
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute -top-32 -right-32 w-[500px] h-[500px] bg-primary/[0.03] rounded-full blur-[150px]" />
        </div>

        <header className="h-14 flex justify-between items-center px-6 border-b border-white/[0.05] bg-surface-lowest/60 backdrop-blur-xl z-40 shrink-0">
          <span className="font-mono text-[11px] text-on-surface-variant/50 uppercase tracking-[0.1em]">
            Usage Dashboard
          </span>
          <button
            onClick={() => { logout(); router.replace("/login/"); }}
            className="btn-ghost p-2 rounded-lg hover:text-error"
          >
            <span className="material-symbols-outlined text-xl">logout</span>
          </button>
        </header>

        <div className="flex-1 overflow-y-auto scroll-hide relative z-10">
          <div className="max-w-4xl mx-auto px-6 py-8 space-y-8">
            <div>
              <h1 className="text-headline-lg text-white mb-1">Usage</h1>
              <p className="text-on-surface-variant/60 text-body-md">
                Track your API usage and token consumption
              </p>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-3 gap-4">
              {[
                { label: "Total Tokens", value: formatTokens(summary.total_tokens), icon: "token" },
                { label: "Total Actions", value: summary.total_actions.toString(), icon: "bolt" },
                { label: "Chat Messages", value: summary.chat_count.toString(), icon: "chat" },
              ].map((stat) => (
                <div
                  key={stat.label}
                  className="rounded-2xl border border-white/[0.06] bg-surface-container/40 p-5"
                >
                  <div className="flex items-center gap-2 mb-3">
                    <span className="material-symbols-outlined text-primary text-lg">{stat.icon}</span>
                    <span className="font-mono text-[10px] uppercase tracking-[0.15em] text-on-surface-variant/50">
                      {stat.label}
                    </span>
                  </div>
                  <div className="text-headline-sm text-white font-mono">{stat.value}</div>
                </div>
              ))}
            </div>

            {/* Token Usage Bar */}
            <div className="rounded-2xl border border-white/[0.06] bg-surface-container/40 p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-body-lg font-semibold text-white">Token Usage</h3>
                  <p className="text-body-sm text-on-surface-variant/50">
                    {formatTokens(summary.total_tokens)} of {formatTokens(summary.token_limit)} ({summary.plan} plan)
                  </p>
                </div>
                <span className="font-mono text-2xl text-primary font-bold">{summary.token_usage_pct}%</span>
              </div>

              {/* Progress bar */}
              <div className="h-3 bg-surface-container-highest/60 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-700 ease-out"
                  style={{
                    width: `${Math.min(summary.token_usage_pct, 100)}%`,
                    background:
                      summary.token_usage_pct > 90
                        ? "linear-gradient(90deg, #f44336, #ff5722)"
                        : summary.token_usage_pct > 70
                        ? "linear-gradient(90deg, #ff9800, #ffc107)"
                        : "linear-gradient(90deg, var(--color-primary), var(--color-tertiary))",
                  }}
                />
              </div>
            </div>

            {/* Daily Chart */}
            {summary.daily.length > 0 && (
              <div className="rounded-2xl border border-white/[0.06] bg-surface-container/40 p-6 space-y-4">
                <h3 className="text-body-lg font-semibold text-white">Daily Usage (Last 30 Days)</h3>
                <div className="flex items-end gap-1 h-32">
                  {summary.daily.map((day) => (
                    <div
                      key={day.date}
                      className="flex-1 flex flex-col items-center gap-1"
                      title={`${day.date}: ${formatTokens(day.tokens)} tokens, ${day.actions} actions`}
                    >
                      <div
                        className="w-full bg-primary/60 rounded-t-sm transition-all duration-300 hover:bg-primary"
                        style={{ height: `${(day.tokens / maxDailyTokens) * 100}%`, minHeight: "2px" }}
                      />
                      <span className="text-[8px] font-mono text-on-surface-variant/30">
                        {day.date.slice(5)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recent Activity */}
            <div className="rounded-2xl border border-white/[0.06] bg-surface-container/40 p-6 space-y-4">
              <h3 className="text-body-lg font-semibold text-white">Recent Activity</h3>
              {history.length === 0 ? (
                <div className="text-center py-8 text-on-surface-variant/40">
                  <span className="material-symbols-outlined text-3xl mb-2 block">history</span>
                  <span className="text-xs font-mono uppercase tracking-wider">No activity yet</span>
                </div>
              ) : (
                <div className="space-y-2">
                  {history.slice(0, 20).map((record) => (
                    <div
                      key={record.id}
                      className="flex items-center justify-between py-2 border-b border-white/[0.03] last:border-0"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-7 h-7 rounded-lg bg-surface-container-high flex items-center justify-center">
                          <span className="material-symbols-outlined text-xs text-on-surface-variant/60">
                            {record.action === "chat" ? "chat" : record.action === "read_file" ? "description" : record.action === "write_file" ? "edit" : "bolt"}
                          </span>
                        </div>
                        <div>
                          <span className="text-sm text-on-surface capitalize">{record.action.replace("_", " ")}</span>
                          <span className="text-[10px] text-on-surface-variant/30 font-mono ml-2">{formatTime(record.created_at)}</span>
                        </div>
                      </div>
                      <span className="text-xs font-mono text-on-surface-variant/50">
                        {formatTokens(record.tokens_used)} tok
                      </span>
                    </div>
                  ))}
                </div>
              )}
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
          <span className="text-on-surface-variant/30 font-mono text-[10px]">Anzar v0.1.0</span>
        </div>
      </main>
    </div>
  );
}
