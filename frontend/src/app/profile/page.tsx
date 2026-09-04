"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";
import { getProfile, updateProfile, updatePassword } from "@/lib/api";

export default function ProfilePage() {
  const { user, loading: authLoading, logout, refreshUser } = useAuth();
  const router = useRouter();

  const [profile, setProfile] = useState({
    name: "",
    email: "",
    plan: "free",
    auth_provider: "",
    created_at: "",
  });
  const [loading, setLoading] = useState(true);

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);
  const [profileSaved, setProfileSaved] = useState(false);
  const [profileError, setProfileError] = useState("");

  const [currentPw, setCurrentPw] = useState("");
  const [newPw, setNewPw] = useState("");
  const [confirmPw, setConfirmPw] = useState("");
  const [savingPw, setSavingPw] = useState(false);
  const [pwSaved, setPwSaved] = useState(false);
  const [pwError, setPwError] = useState("");

  useEffect(() => {
    if (!authLoading && !user) router.replace("/login/");
  }, [user, authLoading, router]);

  useEffect(() => {
    if (token) {
      getProfile()
        .then((p) => {
          setProfile(p);
          setName(p.name);
          setEmail(p.email);
        })
        .catch(console.error)
        .finally(() => setLoading(false));
    }
  }, [user]);

  const handleSaveProfile = async () => {
    setSavingProfile(true);
    setProfileError("");
    setProfileSaved(false);
    try {
      await updateProfile({ name, email });
      setProfileSaved(true);
      await refreshUser();
      setTimeout(() => setProfileSaved(false), 3000);
    } catch (err: any) {
      setProfileError(err.message || "Failed to update profile");
    } finally {
      setSavingProfile(false);
    }
  };

  const handleChangePassword = async () => {
    setSavingPw(true);
    setPwError("");
    setPwSaved(false);
    if (newPw !== confirmPw) {
      setPwError("Passwords do not match");
      setSavingPw(false);
      return;
    }
    if (newPw.length < 6) {
      setPwError("Password must be at least 6 characters");
      setSavingPw(false);
      return;
    }
    try {
      await updatePassword({ current_password: currentPw, new_password: newPw });
      setPwSaved(true);
      setCurrentPw("");
      setNewPw("");
      setConfirmPw("");
      setTimeout(() => setPwSaved(false), 3000);
    } catch (err: any) {
      setPwError(err.message || "Failed to change password");
    } finally {
      setSavingPw(false);
    }
  };

  const token = user ? localStorage.getItem("anzar_token") : null;

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
          <div className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl bg-primary/10 text-primary">
            <span className="material-symbols-outlined text-xl">person</span>
            <span className="text-body-md font-medium">Profile</span>
          </div>
          <button
            onClick={() => router.push("/usage/")}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-on-surface-variant hover:text-on-surface hover:bg-white/[0.03] transition-all duration-150"
          >
            <span className="material-symbols-outlined text-xl">monitoring</span>
            <span className="text-body-md">Usage</span>
          </button>
        </nav>

        <div className="border-t border-white/[0.04] p-3">
          <div className="flex items-center gap-2.5 px-2 py-2">
            <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-xs font-bold text-primary border border-primary/10">
              {user?.name?.charAt(0)?.toUpperCase() || "U"}
            </div>
            <div className="min-w-0">
              <div className="text-sm font-medium text-on-surface truncate">{user?.name || "User"}</div>
              <div className="text-[10px] text-on-surface-variant/50 font-mono capitalize">{profile.plan} Plan</div>
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
            Profile & Security
          </span>
          <button
            onClick={() => { logout(); router.replace("/login/"); }}
            className="btn-ghost p-2 rounded-lg hover:text-error"
          >
            <span className="material-symbols-outlined text-xl">logout</span>
          </button>
        </header>

        <div className="flex-1 overflow-y-auto scroll-hide relative z-10">
          <div className="max-w-3xl mx-auto px-6 py-8 space-y-10">
            {/* Profile Info */}
            <section className="space-y-5">
              <div>
                <h1 className="text-headline-lg text-white mb-1">Profile</h1>
                <p className="text-on-surface-variant/60 text-body-md">
                  Manage your account details
                </p>
              </div>

              <div className="rounded-2xl border border-white/[0.06] bg-surface-container/40 p-6 space-y-5">
                {/* Avatar */}
                <div className="flex items-center gap-4">
                  <div className="w-14 h-14 rounded-2xl bg-primary/10 border border-primary/10 flex items-center justify-center text-xl font-bold text-primary">
                    {user?.name?.charAt(0)?.toUpperCase() || "U"}
                  </div>
                  <div>
                    <div className="text-body-lg font-semibold text-white">{profile.name}</div>
                    <div className="text-body-sm text-on-surface-variant/50">{profile.email}</div>
                    <div className="mt-1 flex items-center gap-2">
                      <span className="text-[9px] font-mono uppercase tracking-[0.15em] text-primary bg-primary/10 px-2 py-0.5 rounded-full capitalize">
                        {profile.plan}
                      </span>
                      {profile.auth_provider && profile.auth_provider !== "local" && (
                        <span className="text-[9px] font-mono uppercase tracking-[0.15em] text-on-surface-variant/50 bg-white/[0.04] px-2 py-0.5 rounded-full">
                          {profile.auth_provider}
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Fields */}
                <div className="space-y-4">
                  <div className="space-y-1.5">
                    <label className="block font-mono text-label-sm text-on-surface-variant/60">NAME</label>
                    <input
                      type="text"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      className="input-field w-full text-sm"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="block font-mono text-label-sm text-on-surface-variant/60">EMAIL</label>
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="input-field w-full text-sm"
                    />
                  </div>
                </div>

                {profileError && (
                  <div className="text-sm text-error bg-error/10 rounded-lg px-4 py-2">{profileError}</div>
                )}

                <div className="flex items-center gap-3 pt-2">
                  {profileSaved && (
                    <span className="text-green-400 font-mono text-xs uppercase tracking-[0.1em] animate-fade-in">
                      Saved
                    </span>
                  )}
                  <button
                    onClick={handleSaveProfile}
                    disabled={savingProfile}
                    className="btn-primary flex items-center gap-2"
                  >
                    {savingProfile ? (
                      <div className="w-4 h-4 border-2 border-on-primary border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <span className="material-symbols-outlined text-lg">check</span>
                    )}
                    {savingProfile ? "Saving..." : "Save Profile"}
                  </button>
                </div>
              </div>
            </section>

            {/* Password */}
            <section className="space-y-5">
              <div>
                <h2 className="text-headline-sm text-white mb-1">Change Password</h2>
                <p className="text-on-surface-variant/60 text-body-sm">
                  {profile.auth_provider && profile.auth_provider !== "local"
                    ? "Your account uses social login. Password change is not available."
                    : "Update your password to keep your account secure"}
                </p>
              </div>

              <div className={`rounded-2xl border border-white/[0.06] bg-surface-container/40 p-6 space-y-4 ${profile.auth_provider && profile.auth_provider !== "local" ? "opacity-50" : ""}`}>
                <div className="space-y-1.5">
                  <label className="block font-mono text-label-sm text-on-surface-variant/60">CURRENT PASSWORD</label>
                  <input
                    type="password"
                    value={currentPw}
                    onChange={(e) => setCurrentPw(e.target.value)}
                    disabled={!!profile.auth_provider && profile.auth_provider !== "local"}
                    className="input-field w-full text-sm"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="block font-mono text-label-sm text-on-surface-variant/60">NEW PASSWORD</label>
                  <input
                    type="password"
                    value={newPw}
                    onChange={(e) => setNewPw(e.target.value)}
                    disabled={!!profile.auth_provider && profile.auth_provider !== "local"}
                    className="input-field w-full text-sm"
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="block font-mono text-label-sm text-on-surface-variant/60">CONFIRM NEW PASSWORD</label>
                  <input
                    type="password"
                    value={confirmPw}
                    onChange={(e) => setConfirmPw(e.target.value)}
                    disabled={!!profile.auth_provider && profile.auth_provider !== "local"}
                    className="input-field w-full text-sm"
                  />
                </div>

                {pwError && (
                  <div className="text-sm text-error bg-error/10 rounded-lg px-4 py-2">{pwError}</div>
                )}

                <div className="flex items-center gap-3 pt-2">
                  {pwSaved && (
                    <span className="text-green-400 font-mono text-xs uppercase tracking-[0.1em] animate-fade-in">
                      Updated
                    </span>
                  )}
                  <button
                    onClick={handleChangePassword}
                    disabled={savingPw || (!!profile.auth_provider && profile.auth_provider !== "local")}
                    className="btn-primary flex items-center gap-2"
                  >
                    {savingPw ? (
                      <div className="w-4 h-4 border-2 border-on-primary border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <span className="material-symbols-outlined text-lg">lock</span>
                    )}
                    {savingPw ? "Updating..." : "Change Password"}
                  </button>
                </div>
              </div>
            </section>

            {/* Danger Zone */}
            <section className="space-y-5">
              <div className="rounded-2xl border border-error/20 bg-error/5 p-6">
                <h3 className="text-body-lg font-semibold text-error mb-2">Danger Zone</h3>
                <p className="text-body-sm text-on-surface-variant/60 mb-4">
                  Sign out of your account on this device.
                </p>
                <button
                  onClick={() => { logout(); router.replace("/login/"); }}
                  className="px-4 py-2 rounded-xl border border-error/30 text-error text-sm hover:bg-error/10 transition-all"
                >
                  Sign Out
                </button>
              </div>
            </section>
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
