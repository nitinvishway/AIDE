import { useState } from "react";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import "./App.css";

function App() {
  const [isDragOver, setIsDragOver] = useState(false);
  const [label, setLabel] = useState("Drop here");

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

    // Files (dragged from Explorer/Finder)
    if (e.dataTransfer.files.length > 0) {
      const files = Array.from(e.dataTransfer.files);
      setLabel(`Got ${files.length} file(s)`);
      files.forEach((f) => console.log("File:", f.name, f.size));
      // Read content if needed:
      // const reader = new FileReader();
      // reader.onload = () => console.log(reader.result);
      // reader.readAsText(files[0]);
      return;
    }

    // Plain text (dragged from browser, editor, etc.)
    const text = e.dataTransfer.getData("text/plain");
    if (text) {
      setLabel(`Got text: ${text.slice(0, 30)}...`);
      console.log("Dropped text:", text);
      return;
    }

    setLabel("Nothing usable dropped");
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) {
      getCurrentWebviewWindow().startDragging();
    }
  };

  return (
    <div
      id="drop-zone"
      className={isDragOver ? "drag-over" : ""}
      onMouseDown={handleMouseDown}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <span>{label}</span>
    </div>
  );
}

export default App;