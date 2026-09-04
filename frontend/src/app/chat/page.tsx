"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";
import {
  getConversations,
  createConversation,
  getConversation,
  deleteConversation,
  createChatWebSocket,
  uploadFiles,
  getProviders,
  getSettings,
  updateSettings,
} from "@/lib/api";

interface Message {
  id?: string;
  role: "user" | "assistant";
  content: string;
}

interface Conversation {
  id: string;
  title: string;
  created_at: string;
}

export default function ChatPage() {
  const { user, token, loading: authLoading, logout } = useAuth();
  const router = useRouter();

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversation, setActiveConversation] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [providers, setProviders] = useState<{ id: string; name: string; models: string[] }[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [showModelPicker, setShowModelPicker] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragCounterRef = useRef(0);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const PROVIDER_COLORS: Record<string, string> = {
    groq: "#f97316",
    openrouter: "#4d8eff",
    gemini: "#22d3ee",
    openai: "#34d399",
    anthropic: "#fbbf24",
    ollama: "#fb7185",
  };

  const MODEL_PROVIDER_KEY = "anzar.model.provider";
  const MODEL_MODEL_KEY = "anzar.model.model";

  const applyModelPreference = useCallback(
    (provs: { id: string; name: string; models: string[] }[], pref: { provider?: string; model?: string }) => {
      if (!pref?.provider) return false;
      const p = provs.find((x) => x.id === pref.provider);
      if (!p) return false;
      const m = pref.model && p.models.includes(pref.model) ? pref.model : p.models[0];
      setSelectedProvider(p.id);
      setSelectedModel(m);
      return true;
    },
    []
  );

  useEffect(() => {
    if (!authLoading && !user) {
      router.replace("/login/");
    }
  }, [user, authLoading]);

  useEffect(() => {
    if (token) {
      getConversations().then(setConversations).catch(console.error);
      getProviders()
        .then(async (data) => {
          setProviders(data.providers);
          if (data.providers.length === 0) return;

          let pref: { provider?: string; model?: string } = {};
          try {
            const cachedProvider = localStorage.getItem(MODEL_PROVIDER_KEY) || undefined;
            const cachedModel = localStorage.getItem(MODEL_MODEL_KEY) || undefined;
            if (cachedProvider) pref = { provider: cachedProvider, model: cachedModel };
          } catch {
            /* localStorage unavailable */
          }

          try {
            const settings = await getSettings();
            if (settings?.provider) {
              pref = { provider: settings.provider, model: settings.model ?? undefined };
            }
          } catch {
            /* settings unavailable — fall back to cache/default */
          }

          if (applyModelPreference(data.providers, pref)) return;
          const groq = data.providers.find((p) => p.id === "groq") || data.providers[0];
          setSelectedProvider(groq.id);
          setSelectedModel(groq.models[0]);
        })
        .catch(console.error);
    }
  }, [token]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-resize textarea
  const resizeTextarea = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 150) + "px";
  }, []);

  const loadConversation = useCallback(async (id: string) => {
    try {
      const data = await getConversation(id);
      setActiveConversation(id);
      setMessages(
        data.messages.map((m) => ({
          id: m.id,
          role: m.role as "user" | "assistant",
          content: m.content,
        }))
      );
    } catch (err) {
      console.error("Failed to load conversation:", err);
    }
  }, []);

  const handleNewChat = useCallback(async () => {
    try {
      const conv = await createConversation();
      setConversations((prev) => [conv, ...prev]);
      setActiveConversation(conv.id);
      setMessages([]);
    } catch (err) {
      console.error("Failed to create conversation:", err);
    }
  }, []);

  const handleDeleteConversation = useCallback(
    async (id: string) => {
      try {
        await deleteConversation(id);
        setConversations((prev) => prev.filter((c) => c.id !== id));
        if (activeConversation === id) {
          setActiveConversation(null);
          setMessages([]);
        }
      } catch (err) {
        console.error("Failed to delete conversation:", err);
      }
    },
    [activeConversation]
  );

  const addFiles = useCallback((newFiles: FileList | File[]) => {
    const arr = Array.from(newFiles);
    setPendingFiles((prev) => [...prev, ...arr]);
  }, []);

  const removeFile = useCallback((index: number) => {
    setPendingFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current++;
    if (e.dataTransfer.types.includes("Files")) setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current--;
    if (dragCounterRef.current === 0) setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current = 0;
    setIsDragging(false);
    if (e.dataTransfer.files.length > 0) addFiles(e.dataTransfer.files);
  }, [addFiles]);

  const handleSend = useCallback(async () => {
    const hasFiles = pendingFiles.length > 0;
    const hasText = input.trim();
    if ((!hasText && !hasFiles) || isStreaming || isUploading || !token) return;

    let uploadInfo = "";
    if (hasFiles) {
      setIsUploading(true);
      try {
        const result = await uploadFiles(pendingFiles);
        const names = result.uploaded.map((f) => `${f.filename} (${f.size_display})`);
        uploadInfo = `I uploaded ${result.total_uploaded} file${result.total_uploaded !== 1 ? "s" : ""}: ${names.join(", ")}`;
        if (result.errors.length > 0) {
          const errNames = result.errors.map((e) => `${e.filename}: ${e.error}`);
          uploadInfo += `\n\nFailed: ${errNames.join("; ")}`;
        }
      } catch (err: any) {
        setIsUploading(false);
        setPendingFiles([]);
        setMessages((prev) => [...prev, { role: "user", content: "(file upload failed)" }, { role: "assistant", content: `Upload error: ${err.message}` }]);
        return;
      }
      setIsUploading(false);
      setPendingFiles([]);
    }

    const userMessage = hasText ? input.trim() : uploadInfo;
    const fullMessage = hasText && hasFiles ? `${uploadInfo}\n\n${input.trim()}` : userMessage;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: fullMessage }]);
    setIsStreaming(true);

    let convId = activeConversation;
    if (!convId) {
      try {
        const conv = await createConversation(userMessage.substring(0, 50));
        convId = conv.id;
        setActiveConversation(convId);
        setConversations((prev) => [conv, ...prev]);
      } catch (err) {
        console.error("Failed to create conversation:", err);
        setIsStreaming(false);
        return;
      }
    }

    const assistantIndex = messages.length + 1;
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

    const ws = createChatWebSocket(
      token,
      (tokenText) => {
        setMessages((prev) => {
          const updated = [...prev];
          updated[assistantIndex] = {
            ...updated[assistantIndex],
            content: updated[assistantIndex].content + tokenText,
          };
          return updated;
        });
      },
      (conversationId) => {
        setIsStreaming(false);
        if (conversationId !== convId) setActiveConversation(conversationId);
        getConversations().then(setConversations).catch(console.error);
      },
      (error) => {
        console.error("WebSocket error:", error);
        setIsStreaming(false);
        setMessages((prev) => {
          const updated = [...prev];
          updated[assistantIndex] = { role: "assistant", content: `Error: ${error}` };
          return updated;
        });
      }
    );

    wsRef.current = ws;
    ws.onopen = () => {
      const payload: Record<string, any> = { type: "chat", message: fullMessage, conversation_id: convId };
      if (selectedModel) payload.model = selectedModel;
      if (selectedProvider) payload.provider = selectedProvider;
      ws.send(JSON.stringify(payload));
    };
    inputRef.current?.focus();
  }, [input, isStreaming, isUploading, token, activeConversation, messages, pendingFiles, selectedModel, selectedProvider]);

  useEffect(() => {
    return () => { wsRef.current?.close(); };
  }, []);

  useEffect(() => {
    if (!showModelPicker) return;
    const handleClick = () => setShowModelPicker(false);
    const handleKey = (e: KeyboardEvent) => { if (e.key === "Escape") setShowModelPicker(false); };
    document.addEventListener("click", handleClick);
    document.addEventListener("keydown", handleKey);
    return () => { document.removeEventListener("click", handleClick); document.removeEventListener("keydown", handleKey); };
  }, [showModelPicker]);

  const formatTime = (dateStr: string) => {
    const diffMs = Date.now() - new Date(dateStr).getTime();
    const mins = Math.floor(diffMs / 60000);
    if (mins < 1) return "now";
    if (mins < 60) return `${mins}m`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h`;
    return `${Math.floor(hrs / 24)}d`;
  };

  if (authLoading) {
    return (
      <div className="h-screen w-full flex items-center justify-center bg-background">
        <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="h-screen w-full flex flex-col overflow-hidden">
      {/* Top Bar */}
      <header className="h-14 flex justify-between items-center px-5 border-b border-white/[0.05] bg-surface-lowest/60 backdrop-blur-xl z-50 shrink-0">
        <div className="flex items-center gap-2.5">
          <img src="/assets/logo.png" alt="Anzar" className="w-8 h-8 rounded-lg object-contain" />
          <span className="font-display text-xl font-bold text-primary tracking-tight">
            Anzar
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="btn-ghost p-2 rounded-lg md:flex hidden"
            title="Toggle sidebar"
          >
            <span className="material-symbols-outlined text-xl">menu</span>
          </button>
          <button
            onClick={() => router.push("/settings/")}
            className="btn-ghost p-2 rounded-lg"
            title="LLM Providers"
          >
            <span className="material-symbols-outlined text-xl">settings</span>
          </button>
          <button
            onClick={() => router.push("/profile/")}
            className="btn-ghost p-2 rounded-lg"
            title="Profile"
          >
            <span className="material-symbols-outlined text-xl">person</span>
          </button>
          <button
            onClick={() => router.push("/usage/")}
            className="btn-ghost p-2 rounded-lg"
            title="Usage"
          >
            <span className="material-symbols-outlined text-xl">monitoring</span>
          </button>
          <button
            onClick={() => { logout(); router.replace("/login/"); }}
            className="btn-ghost p-2 rounded-lg hover:text-error"
            title="Sign out"
          >
            <span className="material-symbols-outlined text-xl">logout</span>
          </button>
        </div>
      </header>

      {/* Workspace */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside
          className={`${
            sidebarOpen ? "flex" : "hidden"
          } md:flex flex-col h-full w-[280px] shrink-0 border-r border-white/[0.05] bg-surface-lowest/40`}
        >
          {/* New Chat */}
          <div className="p-3">
            <button
              onClick={handleNewChat}
              className="btn-primary w-full flex items-center justify-center gap-2 text-sm"
            >
              <span className="material-symbols-outlined text-lg">add</span>
              New Chat
            </button>
          </div>

          {/* Conversation List */}
          <div className="flex-1 overflow-y-auto scroll-hide px-2 py-1">
            {conversations.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-on-surface-variant/40">
                <span className="material-symbols-outlined text-3xl mb-2">chat_bubble_outline</span>
                <span className="text-xs font-mono uppercase tracking-wider">No chats yet</span>
              </div>
            ) : (
              <div className="space-y-0.5">
                {conversations.map((conv) => (
                  <div
                    key={conv.id}
                    onClick={() => loadConversation(conv.id)}
                    className={`group px-3 py-2.5 cursor-pointer rounded-lg transition-all duration-150 ${
                      activeConversation === conv.id
                        ? "bg-primary/10 border-l-2 border-primary"
                        : "hover:bg-white/[0.03] border-l-2 border-transparent"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span
                        className={`text-sm truncate ${
                          activeConversation === conv.id
                            ? "text-primary font-medium"
                            : "text-on-surface-variant"
                        }`}
                      >
                        {conv.title}
                      </span>
                      <button
                        onClick={(e) => { e.stopPropagation(); handleDeleteConversation(conv.id); }}
                        className="opacity-0 group-hover:opacity-100 shrink-0 text-on-surface-variant/40 hover:text-error transition-all"
                      >
                        <span className="material-symbols-outlined text-base">delete</span>
                      </button>
                    </div>
                    <span className="text-[10px] text-on-surface-variant/30 font-mono mt-0.5 block">
                      {formatTime(conv.created_at)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* User */}
          <div className="border-t border-white/[0.04] p-3">
            <div className="flex items-center gap-2.5">
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

        {/* Chat Area */}
        <section
          className="flex-1 flex flex-col bg-background relative overflow-hidden"
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          {/* Ambient bg */}
          <div className="absolute inset-0 pointer-events-none overflow-hidden">
            <div className="absolute -top-32 -right-32 w-[500px] h-[500px] bg-primary/[0.03] rounded-full blur-[150px]" />
            <div className="absolute -bottom-32 -left-32 w-[400px] h-[400px] bg-secondary/[0.02] rounded-full blur-[120px]" />
          </div>

          {/* Drag overlay */}
          {isDragging && (
            <div className="drag-overlay">
              <span className="material-symbols-outlined text-primary text-5xl mb-3">cloud_upload</span>
              <span className="text-headline-sm text-primary font-semibold">Drop files here</span>
              <span className="text-body-sm text-on-surface-variant/60 mt-1">Files will be uploaded to your workspace</span>
            </div>
          )}

          {/* Messages */}
          <div className="flex-1 overflow-y-auto scroll-hide relative z-10">
            {messages.length === 0 ? (
              <div className="h-full flex items-center justify-center">
                <div className="text-center animate-fade-in">
                  <div className="w-14 h-14 mx-auto mb-4 rounded-2xl bg-surface-container/60 border border-white/[0.05] flex items-center justify-center">
                    <span
                      className="material-symbols-outlined text-primary text-3xl"
                      style={{ fontVariationSettings: "'FILL' 1" }}
                    >
                      auto_awesome
                    </span>
                  </div>
                  <h2 className="text-headline-md text-white mb-2">
                    How can I help?
                  </h2>
                  <p className="text-on-surface-variant/60 max-w-sm mx-auto text-body-md">
                    Ask anything about your code, or create a file.
                  </p>
                </div>
              </div>
            ) : (
              <div className="max-w-4xl mx-auto px-6 py-8 space-y-6">
                {messages.map((msg, i) => (
                  <div
                    key={i}
                    className={`flex gap-4 animate-slide-up ${msg.role === "user" ? "flex-row-reverse" : ""}`}
                    style={{ animationDelay: `${Math.min(i * 0.05, 0.3)}s` }}
                  >
                    {/* Avatar */}
                    <div
                      className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
                        msg.role === "user"
                          ? "bg-primary/10 border border-primary/10"
                          : "bg-surface-container-high border border-white/[0.04]"
                      }`}
                    >
                      <span
                        className="material-symbols-outlined text-base text-primary"
                        style={msg.role === "assistant" ? { fontVariationSettings: "'FILL' 1" } : {}}
                      >
                        {msg.role === "user" ? "person" : "auto_awesome"}
                      </span>
                    </div>

                    {/* Content */}
                    <div className={`flex flex-col max-w-[80%] ${msg.role === "user" ? "items-end" : ""}`}>
                      <span className="text-[10px] font-mono text-on-surface-variant/30 uppercase tracking-wider mb-1">
                        {msg.role === "user" ? "You" : "Anzar"}
                      </span>
                      <div
                        className={`px-4 py-3 rounded-2xl text-body-md leading-relaxed ${
                          msg.role === "user"
                            ? "bg-primary/10 text-on-surface border border-primary/[0.08]"
                            : "bg-surface-container/80 text-on-surface-variant border border-white/[0.04]"
                        }`}
                      >
                        {msg.content ? (
                          <p className="whitespace-pre-wrap">{msg.content}</p>
                        ) : (
                          <div className="flex items-center gap-1.5 py-1">
                            <span className="w-1.5 h-1.5 bg-primary/50 rounded-full animate-pulse" />
                            <span className="w-1.5 h-1.5 bg-primary/50 rounded-full animate-pulse" style={{ animationDelay: "0.15s" }} />
                            <span className="w-1.5 h-1.5 bg-primary/50 rounded-full animate-pulse" style={{ animationDelay: "0.3s" }} />
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Input */}
          <div className="relative z-20 p-4 px-6">
            <div className="max-w-4xl mx-auto">
              {/* Input box */}
              <div className="relative group">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-primary/10 via-tertiary/10 to-primary/10 rounded-2xl blur-sm opacity-30 group-focus-within:opacity-70 transition duration-500" />
                <div className="relative bg-surface-lowest/90 backdrop-blur-xl border border-white/[0.06] rounded-2xl overflow-hidden">

                  {/* Pending file chips — inside the box */}
                  {pendingFiles.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 px-4 pt-3 pb-1">
                      {pendingFiles.map((file, i) => (
                        <div
                          key={i}
                          className="upload-chip flex items-center gap-1.5 bg-primary/10 border border-primary/20 rounded-lg px-2.5 py-1 text-xs"
                        >
                          <span className="material-symbols-outlined text-primary text-sm">description</span>
                          <span className="text-on-surface truncate max-w-[120px]">{file.name}</span>
                          <button
                            onClick={() => removeFile(i)}
                            className="text-on-surface-variant/40 hover:text-error transition-colors"
                          >
                            <span className="material-symbols-outlined text-sm">close</span>
                          </button>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Textarea + buttons row */}
                  <div className="flex items-end gap-1 px-3 py-2">
                    {/* Attach */}
                    <input
                      ref={fileInputRef}
                      type="file"
                      multiple
                      className="hidden"
                      onChange={(e) => { if (e.target.files) addFiles(e.target.files); e.target.value = ""; }}
                    />
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      disabled={isStreaming || isUploading}
                      className="text-on-surface-variant/40 hover:text-primary hover:bg-primary/10 transition-all w-9 h-9 rounded-lg flex items-center justify-center shrink-0 disabled:opacity-30 mb-px"
                      title="Attach files"
                    >
                      {isUploading ? (
                        <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                      ) : (
                        <span className="material-symbols-outlined text-xl">attach_file</span>
                      )}
                    </button>

                    {/* Textarea */}
                    <textarea
                      ref={textareaRef}
                      value={input}
                      onChange={(e) => { setInput(e.target.value); resizeTextarea(); }}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          handleSend();
                          requestAnimationFrame(() => { if (textareaRef.current) textareaRef.current.style.height = "auto"; });
                        }
                      }}
                      rows={1}
                      className="chat-textarea flex-1 py-2"
                    />

                    {/* Send */}
                    <button
                      onClick={() => { handleSend(); requestAnimationFrame(() => { if (textareaRef.current) textareaRef.current.style.height = "auto"; }); }}
                      disabled={(!input.trim() && pendingFiles.length === 0) || isStreaming || isUploading}
                      className="bg-primary text-on-primary w-9 h-9 rounded-xl flex items-center justify-center hover:shadow-glow transition-all active:scale-95 disabled:opacity-20 disabled:hover:shadow-none shrink-0 mb-px"
                    >
                      <span className="material-symbols-outlined text-xl">arrow_upward</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* Model toggle row */}
              <div className="flex items-center gap-2 mt-2 ml-1">
                <div className="relative">
                  <div className="model-toggle-row">
                    <span className="model-pill-label">
                      <span className="material-symbols-outlined" style={{ fontSize: "13px" }}>smart_toy</span>
                      Model
                    </span>
                    <button
                      onClick={(e) => { e.stopPropagation(); setShowModelPicker(!showModelPicker); }}
                      disabled={isStreaming}
                      className="model-pill disabled:opacity-30"
                    >
                      <span
                        className="dot"
                        style={{ background: PROVIDER_COLORS[selectedProvider || "groq"] || "#8c909f" }}
                      />
                      <span className="truncate max-w-[140px]">
                        {selectedProvider && selectedModel
                          ? `${providers.find((p) => p.id === selectedProvider)?.name || selectedProvider} · ${selectedModel.split("/").pop()}`
                          : "Select model"}
                      </span>
                      <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>expand_more</span>
                    </button>
                  </div>

                  {showModelPicker && (
                    <div onClick={(e) => e.stopPropagation()} className="model-dropdown absolute bottom-full left-0 mb-2 w-80 bg-surface-lowest/95 backdrop-blur-xl border border-white/[0.08] rounded-2xl shadow-2xl overflow-hidden z-50">
                      <div className="p-3 border-b border-white/[0.05]">
                        <span className="text-label-md text-on-surface-variant/40">Select model</span>
                      </div>
                      <div className="max-h-72 overflow-y-auto scroll-hide p-1.5">
                        {providers.map((provider) => (
                          <div key={provider.id} className="mb-1">
                            <div className="px-2.5 py-1.5 flex items-center gap-2">
                              <span
                                className="w-2 h-2 rounded-full shrink-0"
                                style={{ background: PROVIDER_COLORS[provider.id] || "#8c909f" }}
                              />
                              <span className="text-label-md text-on-surface-variant/50 uppercase tracking-wider">
                                {provider.name}
                              </span>
                            </div>
                            {provider.models.map((model) => {
                              const isActive = selectedProvider === provider.id && selectedModel === model;
                              return (
                                <button
                                  key={model}
                                  onClick={() => {
                                    setSelectedProvider(provider.id);
                                    setSelectedModel(model);
                                    setShowModelPicker(false);
                                    try {
                                      localStorage.setItem(MODEL_PROVIDER_KEY, provider.id);
                                      localStorage.setItem(MODEL_MODEL_KEY, model);
                                    } catch {
                                      /* localStorage unavailable */
                                    }
                                    updateSettings({ provider: provider.id, model }).catch(console.error);
                                  }}
                                  className={`w-full text-left px-3 py-2 text-xs font-mono rounded-lg transition-all flex items-center gap-2.5 ${
                                    isActive
                                      ? "bg-primary/10 text-primary"
                                      : "text-on-surface-variant hover:bg-white/[0.04]"
                                  }`}
                                >
                                  <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                                    isActive ? "bg-primary" : "bg-white/10"
                                  }`} />
                                  <span className="truncate">{model}</span>
                                  {isActive && (
                                    <span className="material-symbols-outlined text-primary ml-auto" style={{ fontSize: "14px" }}>check</span>
                                  )}
                                </button>
                              );
                            })}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
