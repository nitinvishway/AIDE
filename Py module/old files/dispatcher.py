"""
dispatcher.py

Reads the JSONL metadata log produced by file_fetcher.py (sorted ascending
by size, so small/cheap files are processed before big/slow ones) and, for
each entry, runs a type-specific extraction:

    - PDF / documents  -> extract text (key details)
    - images           -> OCR text + image description (pluggable model)
    - video            -> frame-by-frame scene description + audio transcript
    - audio            -> speech-to-text transcript
    - spreadsheets/code -> raw text read

Every result — success or failure — is appended as one JSON line to a
COMMON results log (results_log.jsonl by default), so downstream anything
(search, indexing, another LLM pass) can consume a single unified file
instead of digging through per-type outputs.

Install what you need depending on which handlers you actually use:
    pip install pypdf                      # PDF text extraction (fallback path)
    pip install torch transformers pymupdf # PDF OCR via baidu/Unlimited-OCR (needs NVIDIA GPU)
    pip install pillow pytesseract         # image OCR (+ system tesseract binary)
    pip install opencv-python              # video frame extraction
    pip install openai-whisper             # audio/video transcription
    (ffmpeg must be installed and on PATH for audio extraction from video)

PDF handling uses baidu/Unlimited-OCR (https://huggingface.co/baidu/Unlimited-OCR)
when a GPU + its dependencies are available — it's a 3B-param vision-language OCR
model, much stronger than plain text extraction for scanned/complex PDFs. If it's
unavailable (no GPU, deps missing, or first-time model download fails), the PDF
handler falls back to pypdf's plain text extraction automatically.

Anything not installed degrades gracefully: the handler logs a record with
status="skipped" and a note on what's missing, rather than crashing the run.

Usage:
    from file_fetcher import FileFetcher
    from dispatcher import Dispatcher

    fetcher = FileFetcher(whole_computer=True)
    log_path = fetcher.fetch_to_log("scan_log.jsonl")
    sorted_log = fetcher.sort_log_by_size(log_path, descending=False)  # ascending

    dispatcher = Dispatcher(results_log_path="results_log.jsonl")
    dispatcher.run(sorted_log)
"""
import pypdf
import json
import os
from datetime import datetime
from typing import Callable, Dict, Optional, List


# ---------------------------------------------------------------------------
# Shared results logger — every handler funnels its output through this,
# so there is exactly one common log file downstream consumers read.
# ---------------------------------------------------------------------------
class ResultsLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        # truncate/create fresh at the start of a run
        open(self.log_path, "w", encoding="utf-8").close()

    def log(
        self,
        record: dict,
        status: str,
        description: str = "",
        extra: Optional[dict] = None,
    ) -> None:
        """
        Append one processed-file result to the common log.

        :param record: original metadata record from the scan log
        :param status: "ok" | "error" | "skipped"
        :param description: the extracted/derived textual content
            (OCR text, transcript, extracted PDF text, scene description...)
        :param extra: any handler-specific extra fields (e.g. page_count,
            frame_count, duration_seconds)
        """
        entry = {
            "path": record.get("path"),
            "name": record.get("name"),
            "category": record.get("category"),
            "extension": record.get("extension"),
            "size_bytes": record.get("size_bytes"),
            "status": status,
            "description": description,
            "processed_at": datetime.now().isoformat(),
        }
        if extra:
            entry.update(extra)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Handlers. Each takes (record, logger) and is responsible for calling
# logger.log(...) exactly once with the outcome.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PDF OCR via baidu/Unlimited-OCR — loaded once and reused across calls
# (loading a 3B-param model per file would be extremely slow). Falls back
# to pypdf plain-text extraction if the model/deps/GPU aren't available.
# ---------------------------------------------------------------------------
_unlimited_ocr_model = None
_unlimited_ocr_tokenizer = None


def _load_unlimited_ocr():
    """Lazily load and cache baidu/Unlimited-OCR. Raises on failure."""
    global _unlimited_ocr_model, _unlimited_ocr_tokenizer
    if _unlimited_ocr_model is not None:
        return _unlimited_ocr_model, _unlimited_ocr_tokenizer

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "baidu/Unlimited-OCR"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().cuda()  # requires an NVIDIA GPU

    _unlimited_ocr_model, _unlimited_ocr_tokenizer = model, tokenizer
    return model, tokenizer


def _pdf_to_page_images(pdf_path: str, dpi: int = 300):
    """Render each PDF page to a PNG via PyMuPDF. Returns (paths, tmp_dir)."""
    import fitz  # PyMuPDF
    import tempfile

    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths, tmp_dir


def _ocr_pdf_with_unlimited_ocr(pdf_path: str) -> Optional[str]:
    """
    Run baidu/Unlimited-OCR over every page of a PDF in one pass.
    Returns the parsed text, or None if the model/deps aren't available
    (caller should fall back to pypdf in that case).

    NOTE: the model card doesn't fully document infer_multi's return value
    or the exact output filename(s) written to output_path when
    save_results=True. This reads back whatever text/markdown files land in
    the output dir — verify against your installed version and adjust if
    the actual output layout differs.
    """
    import shutil
    import tempfile
    import glob

    try:
        model, tokenizer = _load_unlimited_ocr()
    except ImportError:
        return None
    except Exception:
        return None  # e.g. no CUDA GPU available

    image_paths, img_tmp_dir = _pdf_to_page_images(pdf_path)
    out_dir = tempfile.mkdtemp(prefix="ocr_out_")

    try:
        model.infer_multi(
            tokenizer,
            prompt="<image>Multi page parsing.",
            image_files=image_paths,
            output_path=out_dir,
            image_size=1024,
            max_length=32768,
            no_repeat_ngram_size=35,
            ngram_window=1024,
            save_results=True,
        )
        # Collect whatever text/markdown output files were written.
        output_files = sorted(
            glob.glob(os.path.join(out_dir, "*.txt")) + glob.glob(os.path.join(out_dir, "*.md"))
        )
        texts = []
        for fp in output_files:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
        return "\n\n".join(texts) if texts else None
    finally:
        shutil.rmtree(img_tmp_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def read_document(record: dict, logger: ResultsLogger) -> None:
    """Extract text from PDFs/docs. PDFs try Unlimited-OCR first, then pypdf."""
    path = record["path"]
    ext = record.get("extension", "")

    if ext != ".pdf":
        # .docx/.txt/etc — plain text formats can just be read directly
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            logger.log(record, "ok", text[:5000], {"char_count": len(text)})
        except Exception as e:
            logger.log(record, "error", str(e))
        return

    # Try baidu/Unlimited-OCR first (better for scanned/complex PDFs)
    try:
        ocr_text = _ocr_pdf_with_unlimited_ocr(path)
    except Exception:
        ocr_text = None

    if ocr_text:
        logger.log(
            record, "ok", ocr_text[:5000],
            {"char_count": len(ocr_text), "extraction_method": "unlimited-ocr"},
        )
        return

    # Fallback: plain text extraction via pypdf (no GPU/model needed)
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.log(
            record, "skipped",
            "No PDF backend available (pip install pypdf, or "
            "torch+transformers+pymupdf with an NVIDIA GPU for Unlimited-OCR)",
        )
        return

    try:
        reader = PdfReader(path)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n".join(pages_text)
        logger.log(
            record, "ok", full_text[:5000],
            {"page_count": len(reader.pages), "char_count": len(full_text),
             "extraction_method": "pypdf"},
        )
    except Exception as e:
        logger.log(record, "error", str(e))


def read_spreadsheet(record: dict, logger: ResultsLogger) -> None:
    path = record["path"]
    try:
        import csv
        if record.get("extension") in (".csv", ".tsv"):
            delim = "\t" if record["extension"] == ".tsv" else ","
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                rows = list(csv.reader(f, delimiter=delim))
            preview = "\n".join(",".join(r) for r in rows[:20])
            logger.log(record, "ok", preview, {"row_count": len(rows)})
        else:
            # .xlsx etc
            try:
                import openpyxl
            except ImportError:
                logger.log(record, "skipped", "openpyxl not installed (pip install openpyxl)")
                return
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            sheet_names = wb.sheetnames
            first_sheet = wb[sheet_names[0]]
            preview_rows = []
            for i, row in enumerate(first_sheet.iter_rows(values_only=True)):
                if i >= 15:
                    break
                preview_rows.append(str(row))
            logger.log(
                record, "ok", "\n".join(preview_rows),
                {"sheet_names": sheet_names},
            )
    except Exception as e:
        logger.log(record, "error", str(e))


def read_presentation(record: dict, logger: ResultsLogger) -> None:
    path = record["path"]
    try:
        from pptx import Presentation
    except ImportError:
        logger.log(record, "skipped", "python-pptx not installed (pip install python-pptx)")
        return

    try:
        prs = Presentation(path)
        texts = []
        slide_count = 0
        for slide in prs.slides:
            slide_count += 1
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    texts.append(shape.text)
        combined = "\n".join(texts)
        logger.log(record, "ok", combined[:5000], {"slide_count": slide_count})
    except Exception as e:
        logger.log(record, "error", str(e))


def _ocr_image(path: str) -> Optional[str]:
    """OCR text extraction. Returns None if libs unavailable."""
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return None
    try:
        return pytesseract.image_to_string(Image.open(path))
    except Exception:
        return None


def describe_image_with_model(path: str) -> Optional[str]:
    """
    PLUG-IN POINT: image captioning / visual understanding.

    Wire this up to whatever vision model or API you want — e.g. a local
    BLIP/LLaVA model, or a call to a vision-capable LLM API (including the
    Anthropic API, if you have a key configured) sending the image and
    asking for a textual description. Returning None means "not available",
    and the caller falls back gracefully.
    """
    return None  # TODO: implement with your chosen vision model/API


def read_image(record: dict, logger: ResultsLogger) -> None:
    path = record["path"]
    ocr_text = _ocr_image(path)
    caption = describe_image_with_model(path)

    parts = []
    if caption:
        parts.append(f"Description: {caption}")
    if ocr_text and ocr_text.strip():
        parts.append(f"OCR text: {ocr_text.strip()}")

    if not parts:
        logger.log(
            record, "skipped",
            "No OCR/captioning backend available (install pillow+pytesseract, "
            "and/or wire up describe_image_with_model)",
        )
        return

    logger.log(record, "ok", "\n".join(parts))


def _transcribe_audio(path: str) -> Optional[str]:
    """
    Speech-to-text. Tries openai-whisper if installed. Returns None if
    unavailable so callers can degrade gracefully.
    """
    try:
        import whisper
    except ImportError:
        return None
    try:
        # Loading the model fresh per call is slow for large batches —
        # for real use, load once and reuse (e.g. cache on the Dispatcher).
        model = whisper.load_model("base")
        result = model.transcribe(path)
        return result.get("text", "")
    except Exception:
        return None


def read_audio(record: dict, logger: ResultsLogger) -> None:
    path = record["path"]
    transcript = _transcribe_audio(path)
    if transcript is None:
        logger.log(
            record, "skipped",
            "No transcription backend available (pip install openai-whisper)",
        )
        return
    logger.log(record, "ok", transcript.strip())


def _extract_video_frames(path: str, every_n_seconds: float = 2.0) -> List:
    """Sample frames from a video at a fixed interval using OpenCV."""
    import cv2
    frames = []
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(int(fps * every_n_seconds), 1)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def _dedupe_frames(frames: List) -> List:
    """
    Drop near-duplicate consecutive frames using a cheap perceptual hash,
    so a static scene doesn't get described dozens of times.
    """
    import cv2

    def phash(frame) -> str:
        small = cv2.resize(frame, (8, 8), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        avg = gray.mean()
        return "".join("1" if p > avg else "0" for p in gray.flatten())

    unique = []
    last_hash = None
    for frame in frames:
        h = phash(frame)
        if h != last_hash:
            unique.append(frame)
            last_hash = h
    return unique


def read_video(record: dict, logger: ResultsLogger) -> None:
    path = record["path"]

    try:
        import cv2  # noqa: F401
    except ImportError:
        logger.log(record, "skipped", "opencv-python not installed (pip install opencv-python)")
        return

    try:
        frames = _extract_video_frames(path, every_n_seconds=2.0)
        unique_frames = _dedupe_frames(frames)
    except Exception as e:
        logger.log(record, "error", f"frame extraction failed: {e}")
        return

    # Describe each unique frame (reuses the same plug-in point as images).
    # Frames are written to a temp file since describe_image_with_model
    # expects a path — swap for an in-memory variant if your model supports it.
    import tempfile
    import cv2 as _cv2

    scene_descriptions = []
    for i, frame in enumerate(unique_frames):
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = tmp.name
        tmp.close()
        _cv2.imwrite(tmp_path, frame)
        caption = describe_image_with_model(tmp_path)
        os.unlink(tmp_path)
        if caption:
            scene_descriptions.append(f"Frame {i}: {caption}")

    # Audio track: extract + transcribe (requires ffmpeg on PATH)
    transcript = None
    audio_path = path + ".__extracted_audio.wav"
    try:
        os.system(f'ffmpeg -y -i "{path}" -ac 1 -ar 16000 "{audio_path}" -loglevel quiet')
        if os.path.exists(audio_path):
            transcript = _transcribe_audio(audio_path)
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)

    parts = []
    if scene_descriptions:
        parts.append("Visual scenes:\n" + "\n".join(scene_descriptions))
    else:
        parts.append(
            f"({len(unique_frames)} unique frames found, but no captioning "
            f"backend wired up — see describe_image_with_model)"
        )
    if transcript:
        parts.append(f"Audio transcript:\n{transcript.strip()}")

    logger.log(
        record, "ok", "\n\n".join(parts),
        {"frame_count_sampled": len(frames), "unique_frame_count": len(unique_frames)},
    )


def read_archive(record: dict, logger: ResultsLogger) -> None:
    path = record["path"]
    try:
        import zipfile, tarfile
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
        elif tarfile.is_tarfile(path):
            with tarfile.open(path) as t:
                names = t.getnames()
        else:
            logger.log(record, "skipped", "unsupported archive format")
            return
        logger.log(record, "ok", "\n".join(names[:200]), {"entry_count": len(names)})
    except Exception as e:
        logger.log(record, "error", str(e))


def read_code(record: dict, logger: ResultsLogger) -> None:
    path = record["path"]
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        logger.log(record, "ok", text[:5000], {"line_count": text.count("\n") + 1})
    except Exception as e:
        logger.log(record, "error", str(e))


def read_ebook(record: dict, logger: ResultsLogger) -> None:
    logger.log(record, "skipped", "ebook text extraction not yet implemented")


def read_other(record: dict, logger: ResultsLogger) -> None:
    logger.log(record, "skipped", "no handler registered for this file type")


# Category -> handler function. Must match categories from file_fetcher.CATEGORY_MAP.
DEFAULT_HANDLERS: Dict[str, Callable[[dict, ResultsLogger], None]] = {
    "document": read_document,
    "spreadsheet": read_spreadsheet,
    "presentation": read_presentation,
    "image": read_image,
    "audio": read_audio,
    "video": read_video,
    "archive": read_archive,
    "code": read_code,
    "ebook": read_ebook,
    "other": read_other,
}


class Dispatcher:
    def __init__(
        self,
        handlers: Optional[Dict[str, Callable[[dict, ResultsLogger], None]]] = None,
        results_log_path: str = "results_log.jsonl",
    ):
        """
        :param handlers: category -> handler(record, logger). Defaults to DEFAULT_HANDLERS.
        :param results_log_path: the single common log every handler writes results to.
        """
        self.handlers = handlers or DEFAULT_HANDLERS
        self.logger = ResultsLogger(results_log_path)

    def dispatch_one(self, record: dict) -> None:
        category = record.get("category", "other")
        handler = self.handlers.get(category, self.handlers.get("other"))
        if handler is None:
            return
        try:
            handler(record, self.logger)
        except Exception as e:
            # last-resort catch so one bad file never kills the whole run
            self.logger.log(record, "error", f"unhandled exception: {e}")

    def run(
        self,
        scan_log_path: str,
        limit: Optional[int] = None,
        progress_every: int = 100,
        progress_callback: Optional[Callable[[int, dict], None]] = None,
    ) -> int:
        """
        Stream the (ascending-sorted, small-files-first) scan log and
        dispatch each record. Doesn't load the whole log into memory.
        """
        processed = 0
        with open(scan_log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                self.dispatch_one(record)

                processed += 1
                if progress_callback and progress_every and processed % progress_every == 0:
                    progress_callback(processed, record)

                if limit and processed >= limit:
                    break

        return processed


if __name__ == "__main__":
    def report(count, record):
        print(f"  ...{count} files processed (last: {record.get('name')})")

    dispatcher = Dispatcher(results_log_path="results_log.jsonl")
    count = dispatcher.run("scan_log.jsonl", progress_every=100, progress_callback=report)
    print(f"\nDispatched {count} files. Results written to results_log.jsonl")