export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  user: {
    id: string;
    email: string;
    name: string;
    plan?: string;
  };
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  metadata_json?: string;
}

export interface Settings {
  provider: string;
  model?: string;
  api_key_masked?: string;
}

const API_BASE = "";

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("anzar_token");
}

async function apiRequest(
  path: string,
  options: RequestInit = {},
  timeoutMs: number = 8000
): Promise<any> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers,
      signal: controller.signal,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Request failed" }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  } finally {
    clearTimeout(timer);
  }
}

// Auth
export async function register(
  name: string,
  email: string,
  password: string
): Promise<AuthTokens> {
  return apiRequest("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ name, email, password }),
  });
}

export async function login(
  email: string,
  password: string
): Promise<AuthTokens> {
  return apiRequest("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function getMe(): Promise<{
  id: string;
  email: string;
  name: string;
  plan: string;
}> {
  return apiRequest("/api/auth/me");
}

export async function getAuthConfig(): Promise<{
  google: boolean;
  google_client_id: string;
  github: boolean;
}> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch(`/api/auth/config`, { signal: controller.signal });
    return response.json();
  } finally {
    clearTimeout(timer);
  }
}

// Conversations
export async function getConversations(): Promise<Conversation[]> {
  return apiRequest("/api/conversations");
}

export async function createConversation(
  title?: string
): Promise<Conversation> {
  const query = title ? `?title=${encodeURIComponent(title)}` : "";
  return apiRequest(`/api/conversations${query}`, { method: "POST" });
}

export async function getConversation(
  id: string
): Promise<{ conversation: Conversation; messages: Message[] }> {
  return apiRequest(`/api/conversations/${id}`);
}

export async function deleteConversation(id: string): Promise<void> {
  return apiRequest(`/api/conversations/${id}`, { method: "DELETE" });
}

// Chat
export async function sendMessage(
  message: string,
  conversationId?: string
): Promise<{ conversation_id: string; response: string }> {
  return apiRequest("/api/chat", {
    method: "POST",
    body: JSON.stringify({ message, conversation_id: conversationId }),
  });
}

// Settings
export async function getSettings(): Promise<Settings> {
  return apiRequest("/api/settings");
}

export async function updateSettings(data: {
  provider?: string;
  api_key?: string;
  model?: string;
}): Promise<Settings> {
  return apiRequest("/api/settings", {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

// Workspace
export async function getWorkspaceStatus(): Promise<any> {
  return apiRequest("/api/workspace/status");
}

// Providers
export async function getProviders(): Promise<{ providers: { id: string; name: string; models: string[] }[] }> {
  return apiRequest("/api/providers");
}

// Upload
export async function uploadFiles(
  files: File[]
): Promise<{ uploaded: { filename: string; size_display: string; path: string }[]; errors: { filename: string; error: string }[]; total_uploaded: number }> {
  const token = getToken();
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(`${API_BASE}/api/workspace/upload`, {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: formData,
      signal: controller.signal,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Upload failed" }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }
    return response.json();
  } finally {
    clearTimeout(timer);
  }
}

// Profile
export async function getProfile(): Promise<{
  id: string;
  email: string;
  name: string;
  plan: string;
  auth_provider: string;
  avatar_url: string | null;
  created_at: string;
}> {
  return apiRequest("/api/profile");
}

export async function updateProfile(data: {
  name?: string;
  email?: string;
}): Promise<{ id: string; email: string; name: string; plan: string }> {
  return apiRequest("/api/profile", {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export async function updatePassword(data: {
  current_password: string;
  new_password: string;
}): Promise<{ status: string; message: string }> {
  return apiRequest("/api/profile/password", {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

// Usage
export async function getUsageSummary(): Promise<{
  total_tokens: number;
  total_actions: number;
  chat_count: number;
  daily: { date: string; tokens: number; actions: number }[];
  plan: string;
  token_limit: number;
  token_usage_pct: number;
}> {
  return apiRequest("/api/usage/summary");
}

export async function getUsageHistory(): Promise<
  { id: string; action: string; tokens_used: number; created_at: string }[]
> {
  return apiRequest("/api/usage/history");
}

// WebSocket
export function createChatWebSocket(
  token: string,
  onToken: (token: string) => void,
  onDone: (conversationId: string) => void,
  onError: (error: string) => void
): WebSocket {
  const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(
    `${wsProtocol}//${window.location.host}/ws/chat?token=${token}`
  );

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "token") {
      onToken(data.content);
    } else if (data.type === "done") {
      onDone(data.conversation_id);
    } else if (data.type === "error") {
      onError(data.message);
    }
  };

  ws.onerror = () => {
    onError("WebSocket connection error");
  };

  return ws;
}
