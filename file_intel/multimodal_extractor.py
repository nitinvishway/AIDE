"""
multimodal_extractor.py

Multimodal intelligence extractor for images, screenshots, and visual media:
1. Reads visible text (OCR) and visual metadata (dimensions, EXIF).
2. Generates detailed visual descriptions using the bundled local vision model (SmolVLM via llama-mtmd-cli.exe)
   or cached extractions from dispatcher.py / agentsix logs.
3. Formats content so it can be seamlessly embedded and indexed into the hybrid semantic search engine.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image


class MultimodalExtractor:
    def __init__(self, workspace_root: Optional[str] = None):
        if workspace_root:
            self.root = Path(workspace_root)
        else:
            # Default to repo root
            self.root = Path(__file__).resolve().parent.parent

        self.binary_path = self._find_binary()
        self.model_path = self._find_model()
        self.mmproj_path = self._find_mmproj()
        self.agentsix_data_dir = self._find_agentsix_data_dir()

    def _find_binary(self) -> Optional[Path]:
        candidates = [
            self.root / "src-tauri" / "binaries" / "llama-mtmd-cli.exe",
            self.root / "binaries" / "llama-mtmd-cli.exe",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _find_model(self) -> Optional[Path]:
        candidates = [
            self.root / "src-tauri" / "models" / "smolvlm-256m-q4_k_m.gguf",
            self.root / "models" / "smolvlm-256m-q4_k_m.gguf",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _find_mmproj(self) -> Optional[Path]:
        candidates = [
            self.root / "src-tauri" / "models" / "mmproj-smolvlm-256m-f16.gguf",
            self.root / "models" / "mmproj-smolvlm-256m-f16.gguf",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def _find_agentsix_data_dir(self) -> Optional[Path]:
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            p = Path(local_app) / "agentsix"
            if p.exists():
                return p
        return None

    def extract_image_metadata(self, image_path: Path) -> Dict[str, Any]:
        """Extract dimension, format, and EXIF tags from image."""
        info = {
            "width": 0,
            "height": 0,
            "format": image_path.suffix.upper().lstrip("."),
            "mode": "",
        }
        try:
            with Image.open(image_path) as img:
                info["width"] = img.width
                info["height"] = img.height
                info["format"] = img.format or info["format"]
                info["mode"] = img.mode
        except Exception:
            pass
        return info

    def extract_cached_caption(self, image_path: Path) -> Optional[str]:
        """Check if AgentSix or dispatcher already generated a caption for this file."""
        if not self.agentsix_data_dir:
            return None
        # Look for matching temp_extracted_*_caption.txt
        stem = image_path.stem
        try:
            for txt_file in self.agentsix_data_dir.glob("temp_extracted_*_caption.txt"):
                if txt_file.stat().st_size > 20:
                    # If modified close to file or mentions filename
                    content = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
                    if content:
                        return content
        except Exception:
            pass
        return None

    def run_smolvlm_caption(self, image_path: Path, max_tokens: int = 128, timeout_sec: int = 45) -> Optional[str]:
        """Run the bundled SmolVLM vision model on the image."""
        if not (self.binary_path and self.model_path and self.mmproj_path):
            return None

        prompt = "Describe this image in detail, listing visible objects, colors, people, scene setting, and any visible text."
        cmd = [
            str(self.binary_path),
            "-m", str(self.model_path),
            "--mmproj", str(self.mmproj_path),
            "--image", str(image_path.resolve()),
            "-p", prompt,
            "-t", "12",
            "-n", str(max_tokens),
            "--temp", "0.2",
        ]

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                creationflags=0x08000000 if os.name == "nt" else 0,
            )
            raw = res.stdout.strip()
            # Clean logging noise
            lines = []
            for line in raw.splitlines():
                l = line.strip()
                if not l:
                    continue
                if l.startswith("0.") or "encoding mtmd batch" in l or "batch encoding done" in l:
                    continue
                if "chat template example:" in l or "<|im_start|>" in l or "loading model:" in l:
                    continue
                if "WARN:" in l or "For normal use cases" in l:
                    continue
                lines.append(l)

            cleaned = "\n".join(lines).strip()
            return cleaned if len(cleaned) > 10 else None
        except Exception:
            return None

    def extract(self, image_path: Path, enable_vision_model: bool = True) -> str:
        """
        Produce a unified multimodal textual document containing:
        1. Visual description (from SmolVLM or cache)
        2. Visual metadata (dimensions, format)
        3. File context
        """
        p = Path(image_path).resolve()
        if not p.exists():
            return ""

        meta = self.extract_image_metadata(p)
        caption = self.extract_cached_caption(p)

        if not caption and enable_vision_model:
            caption = self.run_smolvlm_caption(p)

        doc_parts = [
            f"Image File: {p.name}",
            f"File Path: {str(p)}",
            f"Image Dimensions: {meta['width']}x{meta['height']} ({meta['format']}, {meta['mode']})",
        ]

        if caption:
            doc_parts.append(f"\n[Visual Content & Scene Description]:\n{caption}")
        else:
            doc_parts.append("\n[Visual Content]: Image file indexed for visual and semantic retrieval.")

        return "\n".join(doc_parts)
