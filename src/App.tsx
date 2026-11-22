import { useState, useEffect, useRef } from "react";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import { LogicalSize } from "@tauri-apps/api/dpi";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { openPath } from "@tauri-apps/plugin-opener";
import { MarkdownMessage } from "./MarkdownMessage";
import "./App.css";

interface IngestedItem {
  id: string;
  name: string;
  file_type: string; // "image", "pdf", "word", "excel", "text", "other"
  path: string;
  extracted_text_path?: string;
  data_url?: string;
  size_bytes: number;
  text_preview?: string;
}

interface SearchResultItem {
  chunk_id: string;
  file_name: string;
  file_path: string;
  category: string;
  snippet: string;
  final_score: number;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  searchResults?: SearchResultItem[];
}

interface MemoryEntry {
  timestamp: string;
  name: string;
  path: string;
  summary: string;
  keywords: string[];
  text_preview?: string;
}

function formatFileSize(bytes: number): string {
  if (!bytes || bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export default function App() {
  const [isDragOver, setIsDragOver] = useState(false);
  const [activeItem, setActiveItem] = useState<IngestedItem | null>(null);
  const [showChat, setShowChat] = useState(false);
  const [aiPrompt, setAiPrompt] = useState("");
  const [isAiLoading, setIsAiLoading] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [savedMemory, setSavedMemory] = useState<MemoryEntry | null>(null);
  const [themeMode, setThemeMode] = useState<"dark" | "glass">(() => {
    const saved = localStorage.getItem("agentsix_theme_mode");
    return saved === "dark" || saved === "glass" ? saved : "dark";
  });
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [isThinkMode, setIsThinkMode] = useState(false);
  const [mouseDownInfo, setMouseDownInfo] = useState<{ screenX: number; screenY: number } | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const dialogueHistoryRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleAttachClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    fileInputRef.current?.click();
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = "";
    setChatMessages([]);
    setSavedMemory(null);

    const filePath = (file as any).path;
    if (filePath) {
      invoke<IngestedItem>("ingest_file", { path: filePath })
        .then((item) => {
          setActiveItem(item);
          setToastMessage(`📎 Attached ${item.name}`);
          setTimeout(() => setToastMessage(null), 1500);
        })
        .catch((err) => {
          console.error("Failed to ingest file:", err);
        });
    } else {
      const reader = new FileReader();
      reader.onload = () => {
        const resultStr = reader.result as string;
        const base64Data = resultStr.includes(",") ? resultStr.split(",")[1] : resultStr;
        invoke<IngestedItem>("ingest_file_bytes", { name: file.name, base64Data })
          .then((item) => {
            setActiveItem(item);
            setToastMessage(`📎 Attached ${item.name}`);
            setTimeout(() => setToastMessage(null), 1500);
          })
          .catch((err) => {
            console.error("Failed to ingest file bytes:", err);
          });
      };
      reader.readAsDataURL(file);
    }
  };

  const handleToggleThink = (e: React.MouseEvent) => {
    e.stopPropagation();
    const nextMode = !isThinkMode;
    setIsThinkMode(nextMode);
    setToastMessage(nextMode ? "🧠 Think Mode: ON" : "⚡ Fast Mode");
    setTimeout(() => setToastMessage(null), 1400);
    if (nextMode && !showChat) {
      setShowChat(true);
    }
  };

  const handleToggleTheme = (e: React.MouseEvent) => {
    e.stopPropagation();
    const nextMode = themeMode === "dark" ? "glass" : "dark";
    setThemeMode(nextMode);
    localStorage.setItem("agentsix_theme_mode", nextMode);
    setToastMessage(nextMode === "glass" ? "✨ Glass Mode" : "🌙 Dark Mode");
    setTimeout(() => {
      setToastMessage(null);
    }, 1400);
  };

  // Dynamic window height resizing based on chat visibility (strictly hardcoded width 175)
  useEffect(() => {
    const win = getCurrentWebviewWindow();
    const targetHeight = showChat ? 415 : 215;
    win.setSize(new LogicalSize(175, targetHeight)).catch((err) => {
      console.warn("Failed to set dynamic window size:", err);
    });
  }, [showChat]);

  useEffect(() => {
    // 1. Listen for background auto-captured screenshots
    const unlistenScreenshot = listen<IngestedItem>("new-clipboard-image", (event) => {
      const item = event.payload;
      setActiveItem(item);
      setChatMessages([]);
      setSavedMemory(null);
    });

    // 2. Native Tauri window drag-and-drop listener (gives exact Windows paths)
    const unlistenDropPromise = getCurrentWebviewWindow().onDragDropEvent((event) => {
      if (event.payload.type === "enter" || event.payload.type === "over") {
        setIsDragOver(true);
      } else if (event.payload.type === "drop") {
        setIsDragOver(false);
        setChatMessages([]);
        setSavedMemory(null);
        const paths = event.payload.paths;
        if (paths && paths.length > 0) {
          const filePath = paths[0];
          invoke<IngestedItem>("ingest_file", { path: filePath })
            .then((item) => {
              setActiveItem(item);
            })
            .catch((err) => {
              console.error("Failed to ingest dropped file:", err);
            });
        }
      } else if (event.payload.type === "leave") {
        setIsDragOver(false);
      }
    });

    // 3. Manual paste listener (fallback)
    const handleGlobalPaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf("image") !== -1) {
          invoke<IngestedItem>("capture_screenshot")
            .then((res) => {
              setActiveItem(res);
              setChatMessages([]);
              setSavedMemory(null);
            })
            .catch((err) => console.error("Capture screenshot error:", err));
        }
      }
    };

    window.addEventListener("paste", handleGlobalPaste);
    return () => {
      window.removeEventListener("paste", handleGlobalPaste);
      unlistenScreenshot.then((unlisten) => unlisten());
      unlistenDropPromise.then((unlisten) => unlisten());
    };
  }, []);

  useEffect(() => {
    if (dialogueHistoryRef.current) {
      dialogueHistoryRef.current.scrollTop = dialogueHistoryRef.current.scrollHeight;
    }
  }, [chatMessages, savedMemory, isAiLoading]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    setChatMessages([]);
    setSavedMemory(null);

    // 1. Files dropped via HTML5
    if (e.dataTransfer.files.length > 0) {
      const files = Array.from(e.dataTransfer.files);
      const file = files[0];
      const filePath = (file as any).path;

      if (filePath) {
        invoke<IngestedItem>("ingest_file", { path: filePath })
          .then((item) => {
            setActiveItem(item);
          })
          .catch((err) => {
            console.error("Failed to ingest file:", err);
          });
      } else {
        const reader = new FileReader();
        reader.onload = () => {
          const resultStr = reader.result as string;
          const base64Data = resultStr.includes(",") ? resultStr.split(",")[1] : resultStr;
          invoke<IngestedItem>("ingest_file_bytes", { name: file.name, base64Data })
            .then((item) => {
              setActiveItem(item);
            })
            .catch((err) => {
              console.error("Failed to ingest file bytes:", err);
            });
        };
        reader.readAsDataURL(file);
      }
      return;
    }

    // 2. Plain text dropped
    const text = e.dataTransfer.getData("text/plain");
    if (text) {
      invoke<IngestedItem>("ingest_text", { text })
        .then((item) => {
          setActiveItem(item);
        })
        .catch((err) => {
          console.error("Failed to ingest text:", err);
        });
    }
  };

  // Action Button 2: Share
  const handleShare = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!activeItem) {
      return;
    }
    invoke("share_item", { path: activeItem.path })
      .catch(() => {
        navigator.clipboard.writeText(activeItem.path);
      });
  };

  // Action Button 4: Chat Toggle
  const handleToggleChat = (e: React.MouseEvent) => {
    e.stopPropagation();
    setShowChat((prev) => !prev);
  };

  // Multi-turn Dialogue Runner with Google Gemma 2
  const runGemmaDialogue = (promptText: string, targetItem?: IngestedItem | null) => {
    setIsAiLoading(true);
    setShowChat(true);

    const itemToUse = targetItem !== undefined ? targetItem : activeItem;

    let effectivePrompt = promptText;
    if (isThinkMode) {
      effectivePrompt = `[DEEP REASONING INSTRUCTION]: Perform deep step-by-step chain-of-thought thinking before answering. First break down the problem and document context under a '### 🧠 Thinking Process' heading. Then provide your comprehensive structured conclusion under '### 💡 Conclusion'.\n\nPrompt: ${promptText}`;
    }

    // Build conversation history context
    const historyContext = chatMessages
      .map((msg) => `${msg.role === "user" ? "User" : "Assistant"}: ${msg.text}`)
      .join("\n\n");

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      text: promptText,
    };
    setChatMessages((prev) => [...prev, userMsg]);

    invoke<string>("ask_gemma", {
      prompt: effectivePrompt,
      context: historyContext || null,
      filePath: itemToUse?.path || null,
      extractedTextPath: itemToUse?.extracted_text_path || null,
    })
      .then((response) => {
        setIsAiLoading(false);
        const assistantMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          text: response,
        };
        setChatMessages((prev) => [...prev, assistantMsg]);
      })
      .catch((err) => {
        setIsAiLoading(false);
        const errorMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          text: `Gemma 2 info: ${err}`,
        };
        setChatMessages((prev) => [...prev, errorMsg]);
      });
  };

  const handleOpenFile = (filePath: string) => {
    invoke<string>("open_file_in_default_app", { path: filePath })
      .then((res) => {
        console.log("File opened successfully:", res);
        setToastMessage("📂 Opened file");
        setTimeout(() => setToastMessage(null), 1500);
      })
      .catch((err) => {
        console.warn("invoke open_file_in_default_app failed, trying openPath fallback:", err);
        openPath(filePath)
          .then(() => {
            setToastMessage("📂 Opened file");
            setTimeout(() => setToastMessage(null), 1500);
          })
          .catch((e2) => {
            console.error("Failed to open file:", e2 || err);
            setToastMessage("⚠️ Could not open file");
            setTimeout(() => setToastMessage(null), 1500);
          });
      });
  };

  const handleAttachSearchResult = (filePath: string) => {
    invoke<IngestedItem>("ingest_file", { path: filePath })
      .then((item) => {
        setActiveItem(item);
        setToastMessage(`📎 Attached ${item.name}`);
        setTimeout(() => setToastMessage(null), 1500);
      })
      .catch((err) => {
        console.error("Failed to attach search result:", err);
      });
  };

  const runFileSearch = async (queryText: string) => {
    setIsAiLoading(true);
    setShowChat(true);

    const cleanQuery = queryText.replace(/^(find|search|where is|locate|show me|\/search|\/find)\s+/i, "").trim() || queryText;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      text: queryText,
    };
    setChatMessages((prev) => [...prev, userMsg]);

    try {
      const res = await fetch(`http://127.0.0.1:8000/api/search?q=${encodeURIComponent(cleanQuery)}&top_k=4`);
      if (!res.ok) {
        throw new Error(`Search error (${res.status})`);
      }
      const data: SearchResultItem[] = await res.json();
      setIsAiLoading(false);

      if (!data || data.length === 0) {
        const noResultsMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          text: `🔍 No files or media matched "${cleanQuery}". You can attach or drop files to index them!`,
        };
        setChatMessages((prev) => [...prev, noResultsMsg]);
      } else {
        const resultsMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          text: `🔍 Found **${data.length}** matching ${data.length === 1 ? "file" : "files"} for "${cleanQuery}":`,
          searchResults: data,
        };
        setChatMessages((prev) => [...prev, resultsMsg]);
      }
    } catch (err) {
      console.warn("Search API fallback to Gemma dialogue:", err);
      invoke<string>("ask_gemma", {
        prompt: `The user asked: "${queryText}". Answer helpfully.`,
        context: null,
        filePath: activeItem?.path || null,
        extractedTextPath: activeItem?.extracted_text_path || null,
      })
        .then((resp) => {
          setIsAiLoading(false);
          setChatMessages((prev) => [
            ...prev,
            { id: (Date.now() + 1).toString(), role: "assistant", text: resp },
          ]);
        })
        .catch((e2) => {
          setIsAiLoading(false);
          setChatMessages((prev) => [
            ...prev,
            { id: (Date.now() + 1).toString(), role: "assistant", text: `Search notice: ${e2 || err}` },
          ]);
        });
    }
  };

  const handleAiSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!aiPrompt.trim() || isAiLoading) return;
    const promptText = aiPrompt.trim();
    setAiPrompt("");

    const isExplicitSearch = /^(find|search|where is|locate|show me|\/search|\/find)\b/i.test(promptText);
    if (isExplicitSearch || (!activeItem && promptText.length > 2)) {
      runFileSearch(promptText);
    } else {
      runGemmaDialogue(promptText);
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) {
      setMouseDownInfo({
        screenX: e.screenX,
        screenY: e.screenY,
      });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (mouseDownInfo) {
      const distance = Math.hypot(e.screenX - mouseDownInfo.screenX, e.screenY - mouseDownInfo.screenY);
      if (distance > 4) {
        setMouseDownInfo(null);
        getCurrentWebviewWindow().startDragging();
      }
    }
  };

  const handleMouseUp = () => {
    setMouseDownInfo(null);
  };

  return (
    <div
      id="app-container"
      className={`theme-${themeMode}`}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      {/* Toast Notification for Theme Switch / Actions */}
      {toastMessage && (
        <div className="theme-toast-pill">
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Hidden File Picker Input for Attach */}
      <input
        ref={fileInputRef}
        type="file"
        style={{ display: "none" }}
        onChange={handleFileSelect}
      />

      {/* Top Action Bar matching user mockup */}
      <div className="top-action-bar">
        {/* Button 1: Attach File */}
        <button
          className="action-btn"
          onClick={handleAttachClick}
          title="Attach File (select PDF, image, text, code, document)"
          aria-label="Attach File"
        >
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
          </svg>
        </button>

        {/* Button 2: Share */}
        <button
          className="action-btn"
          onClick={handleShare}
          title="Share (copy path)"
          aria-label="Share"
        >
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="18" cy="5" r="3" />
            <circle cx="6" cy="12" r="3" />
            <circle cx="18" cy="19" r="3" />
            <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
            <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
          </svg>
        </button>

        {/* Button 3: Think / Deep Reasoning Mode */}
        <button
          className={`action-btn ${isThinkMode ? "active think-active" : ""}`}
          onClick={handleToggleThink}
          title={isThinkMode ? "Think Mode: ON (Deep Chain-of-Thought Reasoning)" : "Think Mode: OFF (Click to Enable Deep Reasoning)"}
          aria-label="Toggle Think Mode"
        >
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z" />
            <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04z" />
          </svg>
        </button>

        {/* Button 4: Chat Button (Toggles Middle Chat Section) */}
        <button
          className={`action-btn ${showChat ? "active" : ""}`}
          onClick={handleToggleChat}
          title="Toggle Document Chat"
          aria-label="Toggle Document Chat"
        >
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        </button>

        {/* Button 5: Theme Mode Switcher (Glass Mode / Dark Mode) */}
        <button
          className={`action-btn theme-toggle-btn ${themeMode === "glass" ? "glass-mode" : "dark-mode"}`}
          onClick={handleToggleTheme}
          title={`Theme: ${themeMode === "glass" ? "Glass Mode (Click for Dark Mode)" : "Dark Mode (Click for Glass Mode)"}`}
          aria-label="Toggle Glass and Dark Mode"
        >
          {themeMode === "glass" ? (
            <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2L2 7l10 5 10-5-10-5z" />
              <path d="M2 17l10 5 10-5" />
              <path d="M2 12l10 5 10-5" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
            </svg>
          )}
        </button>
      </div>

      {/* Middle Chat Section (Dedicated to Document Dialogue with Gemma 2) */}
      {showChat && (
        <div className="mid-chat-section" onClick={(e) => e.stopPropagation()}>
          <div className="mid-chat-header">
            <div className="ai-tag">
              <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M12 2l2.4 7.2L21 12l-6.6 2.8L12 22l-2.4-7.2L3 12l6.6-2.8L12 2z" />
              </svg>
              <span>Chat</span>
            </div>
            <div className="chat-header-actions">
              <button
                className="chat-theme-btn"
                onClick={handleToggleTheme}
                title={`Switch to ${themeMode === "glass" ? "Dark" : "Glass"} Mode`}
              >
                {themeMode === "glass" ? "✨ Glass" : "🌙 Dark"}
              </button>
              {(chatMessages.length > 0 || savedMemory) && (
                <button
                  className="chat-clear-btn"
                  onClick={() => {
                    setChatMessages([]);
                    setSavedMemory(null);
                  }}
                  title="Clear conversation"
                >
                  Clear
                </button>
              )}
            </div>
          </div>

          {/* Dialogue History Stream (Scrollable) */}
          <div ref={dialogueHistoryRef} className="dialogue-history">
            {/* Display Saved Memory Keywords & Summary inside scroll area */}
            {savedMemory && (
              <div className="memory-card">
                <p className="memory-summary">{savedMemory.summary}</p>
                <div className="memory-keywords">
                  {savedMemory.keywords.map((kw, i) => (
                    <span key={i} className="keyword-pill">#{kw}</span>
                  ))}
                </div>
              </div>
            )}

            {chatMessages.length === 0 && !savedMemory && (
              <div className="chat-placeholder">
                <span>{activeItem ? `Ask anything about ${activeItem.name}` : "Search media or ask Gemma 2"}</span>
              </div>
            )}
            {chatMessages.map((msg) => (
              <div key={msg.id} className={`dialogue-bubble ${msg.role}`}>
                {msg.role === "assistant" ? (
                  <>
                    <MarkdownMessage content={msg.text} />
                    {msg.searchResults && msg.searchResults.length > 0 && (
                      <div className="search-results-tray">
                        {msg.searchResults.map((hit) => (
                          <div key={hit.chunk_id} className="search-hit-card">
                            <div className="search-hit-top">
                              <span className="search-hit-badge">
                                {hit.category === "image" ? "🖼️" : hit.category === "pdf" ? "📄" : "📁"} {hit.category.toUpperCase()}
                              </span>
                              <span className="search-hit-title" title={hit.file_path}>
                                {hit.file_name}
                              </span>
                            </div>
                            <p className="search-hit-snippet" title={hit.snippet}>
                              {hit.snippet.slice(0, 95)}...
                            </p>
                            <div className="search-hit-actions">
                              <button
                                type="button"
                                className="search-hit-btn open"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleOpenFile(hit.file_path);
                                }}
                                title="Open file on your computer"
                              >
                                Open
                              </button>
                              <button
                                type="button"
                                className="search-hit-btn attach"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleAttachSearchResult(hit.file_path);
                                }}
                                title="Attach to widget preview"
                              >
                                Attach
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                ) : (
                  <span>{msg.text}</span>
                )}
              </div>
            ))}
            {isAiLoading && (
              <div className="dialogue-bubble assistant loading">
                <div className="typing-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Input Prompt Box for Continuous Dialogue & Search */}
          <form className="ai-input-bar" onSubmit={handleAiSubmit}>
            <input
              type="text"
              placeholder={
                activeItem
                  ? `Ask about ${activeItem.name}...`
                  : "Search media or ask Gemma..."
              }
              value={aiPrompt}
              onChange={(e) => setAiPrompt(e.target.value)}
            />
            <button
              type="submit"
              className="ai-send-btn"
              disabled={isAiLoading || !aiPrompt.trim()}
              title="Send message"
            >
              <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </form>
        </div>
      )}

      {/* Bottom Drop & Preview Card Zone */}
      <div
        id="drop-zone"
        className={`bottom-preview-card ${isDragOver ? "drag-over" : ""} ${activeItem ? "has-item" : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {!activeItem && (
          <div className="empty-state">
            <span className="empty-label">Drop here</span>
          </div>
        )}

        {activeItem && activeItem.file_type === "image" && (
          <div
            className="preview-image-container"
            onDoubleClick={() => handleOpenFile(activeItem.path)}
            title="Double-click to open in default app"
          >
            {activeItem.data_url ? (
              <img
                src={activeItem.data_url}
                alt={activeItem.name}
                className="preview-image"
              />
            ) : (
              <div className="file-icon-placeholder image-placeholder">
                <svg viewBox="0 0 24 24" width="36" height="36" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                  <circle cx="8.5" cy="8.5" r="1.5" />
                  <polyline points="21 15 16 10 5 21" />
                </svg>
              </div>
            )}
            <div
              className="preview-bottom-pill clickable"
              title={`${activeItem.text_preview || activeItem.name} (Click or double-click to open)`}
              onClick={(e) => {
                e.stopPropagation();
                handleOpenFile(activeItem.path);
              }}
            >
              {activeItem.name === "Live Screen Capture" ? (
                <span className="live-screen-pill-badge">
                  <span className="live-dot"></span> LIVE SCREEN
                </span>
              ) : (
                <span className="window-pill-badge">
                  <span className="live-dot"></span> {activeItem.name.length > 20 ? activeItem.name.slice(0, 18) + "..." : activeItem.name}
                </span>
              )}
            </div>
          </div>
        )}

        {activeItem && activeItem.file_type === "pdf" && (
          <div
            className="preview-file-card pdf-card"
            onDoubleClick={() => handleOpenFile(activeItem.path)}
            onClick={() => handleOpenFile(activeItem.path)}
            title="Click to open in default app"
          >
            <div className="card-icon-badge pdf-badge">
              <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="9" y1="15" x2="15" y2="15" />
              </svg>
              <span className="type-tag">PDF</span>
            </div>
            <div className="file-info">
              <span className="file-name" title={activeItem.name}>{activeItem.name}</span>
              {activeItem.text_preview ? (
                <p className="file-text-snippet">{activeItem.text_preview.slice(0, 70)}...</p>
              ) : (
                <span className="file-size">{formatFileSize(activeItem.size_bytes)}</span>
              )}
            </div>
          </div>
        )}

        {activeItem && activeItem.file_type === "word" && (
          <div
            className="preview-file-card word-card"
            onDoubleClick={() => handleOpenFile(activeItem.path)}
            onClick={() => handleOpenFile(activeItem.path)}
            title="Click to open in default app"
          >
            <div className="card-icon-badge word-badge">
              <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
              </svg>
              <span className="type-tag">DOC</span>
            </div>
            <div className="file-info">
              <span className="file-name" title={activeItem.name}>{activeItem.name}</span>
              <span className="file-size">{formatFileSize(activeItem.size_bytes)}</span>
            </div>
          </div>
        )}

        {activeItem && activeItem.file_type === "excel" && (
          <div
            className="preview-file-card excel-card"
            onDoubleClick={() => handleOpenFile(activeItem.path)}
            onClick={() => handleOpenFile(activeItem.path)}
            title="Click to open in default app"
          >
            <div className="card-icon-badge excel-badge">
              <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="8" y1="12" x2="16" y2="12" />
                <line x1="8" y1="16" x2="16" y2="16" />
                <line x1="12" y1="8" x2="12" y2="16" />
              </svg>
              <span className="type-tag">XLS</span>
            </div>
            <div className="file-info">
              <span className="file-name" title={activeItem.name}>{activeItem.name}</span>
              <span className="file-size">{formatFileSize(activeItem.size_bytes)}</span>
            </div>
          </div>
        )}

        {activeItem && activeItem.file_type === "text" && (
          <div
            className="preview-file-card text-card"
            onDoubleClick={() => handleOpenFile(activeItem.path)}
            onClick={() => handleOpenFile(activeItem.path)}
            title="Click to open in default app"
          >
            <div className="card-icon-badge text-badge">
              <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
                <line x1="10" y1="9" x2="8" y2="9" />
              </svg>
              <span className="type-tag">TXT</span>
            </div>
            <div className="file-info">
              <span className="file-name" title={activeItem.name}>{activeItem.name}</span>
              {activeItem.text_preview ? (
                <p className="file-text-snippet">{activeItem.text_preview.slice(0, 70)}...</p>
              ) : (
                <span className="file-size">{formatFileSize(activeItem.size_bytes)}</span>
              )}
            </div>
          </div>
        )}

        {activeItem && activeItem.file_type === "other" && (
          <div
            className="preview-file-card other-card"
            onDoubleClick={() => handleOpenFile(activeItem.path)}
            onClick={() => handleOpenFile(activeItem.path)}
            title="Click to open in default app"
          >
            <div className="card-icon-badge other-badge">
              <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
              <span className="type-tag">FILE</span>
            </div>
            <div className="file-info">
              <span className="file-name" title={activeItem.name}>{activeItem.name}</span>
              <span className="file-size">{formatFileSize(activeItem.size_bytes)}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}


