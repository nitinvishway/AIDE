# agentsix - Drag & Drop Widget

A lightweight, frameless, and transparent desktop utility built with **Tauri v2**, **React**, and **TypeScript**.

---

## Key Features

- **Borderless & Transparent Window**: Clean styling with custom 20px rounded corners.
- **HTML5 Drag and Drop**: Front-end file and text drops using standard HTML5 APIs (native Tauri drag-and-drop interception is disabled).
- **Window Durability Controls**:
  - **Always-on-top Reassertion**: A background thread guarantees the widget stays on top of other windows.
  - **Show-Desktop Mitigation**: Minimization events (e.g. from pressing `Win + D` or clicking "Show Desktop") are automatically detected and undone within 300ms.

---

## Prerequisites

Ensure you have the following installed on your development machine:
1. **Node.js** (v18+)
2. **Rust** and `cargo` toolchain
3. **WebView2 Runtime** (installed by default on Windows 10/11)

---

## Setup & Running Locally

Follow these steps to run the application in development mode:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/nitinvishway/AIDE.git
   cd AIDE
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Start the Tauri Dev Server**
   ```bash
   npm run tauri dev
   ```
   *This launches the React front-end dev server, compiles the Rust backend, and opens the application.*

---

## Production Build

To build the standalone executable package:
```bash
npm run tauri build
```
The compiled binaries will be located in:
`src-tauri/target/release/` or `src-tauri/target/release/bundle/`

---

## Configuration & Code Architecture

### 1. Window Styling & Properties
Configured in `src-tauri/tauri.conf.json`:
- `"decorations": false` & `"transparent": true` create the borderless canvas.
- `"shadow": false` disables the native Windows 11 shadow/border.
- `"dragDropEnabled": false` enables custom HTML5 drag-and-drop handling on the front-end.

### 2. Front-end Drag & Drop Logic
Found in `src/App.tsx`:
- Tracks drag states via React's state management.
- Handles browser-compatible drops for both files and plain text.

### 3. Rust Backend Loop Control
Found in `src-tauri/src/lib.rs`:
- Uses `.setup` hooks to spawn background threads for checking/reasserting window focus and unminimized states.
