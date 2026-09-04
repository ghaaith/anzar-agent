"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
  useRef,
  ReactNode,
} from "react";
import { getMe } from "@/lib/api";

interface User {
  id: string;
  email: string;
  name: string;
  plan?: string;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  loading: boolean;
  setAuth: (token: string, refreshToken: string, user: User) => void;
  logout: () => void;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const refreshInFlight = useRef(false);

  const refreshUser = useCallback(async () => {
    if (refreshInFlight.current) return;
    refreshInFlight.current = true;
    try {
      const storedToken = localStorage.getItem("anzar_token");
      if (!storedToken) {
        setLoading(false);
        return;
      }
      const me = await getMe();
      setUser(me);
      setToken(storedToken);
    } catch {
      localStorage.removeItem("anzar_token");
      localStorage.removeItem("anzar_refresh_token");
      setUser(null);
      setToken(null);
    } finally {
      setLoading(false);
      refreshInFlight.current = false;
    }
  }, []);

  useEffect(() => {
    refreshUser();
    // Safety net: if loading never resolves, force it off after 10s
    const safety = setTimeout(() => setLoading(false), 10000);
    return () => clearTimeout(safety);
  }, [refreshUser]);

  const setAuth = useCallback(
    (accessToken: string, refreshToken: string, userData: User) => {
      localStorage.setItem("anzar_token", accessToken);
      localStorage.setItem("anzar_refresh_token", refreshToken);
      setToken(accessToken);
      setUser(userData);
    },
    []
  );

  const logout = useCallback(() => {
    localStorage.removeItem("anzar_token");
    localStorage.removeItem("anzar_refresh_token");
    setToken(null);
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({ user, token, loading, setAuth, logout, refreshUser }),
    [user, token, loading, setAuth, logout, refreshUser]
  );

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
