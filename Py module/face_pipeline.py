"""
Google Photos-style face detection, embedding, and clustering pipeline.

Pipeline: detect faces -> align/crop -> generate embeddings -> incrementally
assign to clusters (people) -> periodic batch re-clustering to clean up drift.

Dependencies:
    pip install mediapipe onnxruntime numpy scikit-learn fastapi uvicorn pillow

Models needed (download once, see download_models() below):
    - blaze_face_short_range.tflite   (MediaPipe face detector)
    - mobilefacenet.onnx              (face embedding model, 112x112 input, 128-d output)
"""

import io
import os
import uuid
import sqlite3
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
import onnxruntime as ort
from sklearn.cluster import DBSCAN

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DETECTOR_MODEL_PATH = os.path.join(MODEL_DIR, "blaze_face_short_range.tflite")
EMBEDDER_MODEL_PATH = os.path.join(MODEL_DIR, "mobilefacenet.onnx")
DB_PATH = os.path.join(os.path.dirname(__file__), "faces.db")

DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

CLUSTER_THRESHOLD = 0.55   # euclidean distance threshold for "same person"
MIN_DETECTION_CONFIDENCE = 0.6
EMBEDDING_INPUT_SIZE = 112  # standard MobileFaceNet input


def download_models() -> None:
    """Download the face detector model if not already present.
    MobileFaceNet .onnx must be sourced separately (e.g. exported from a
    pretrained checkpoint) since it isn't hosted by Google."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(DETECTOR_MODEL_PATH):
        print("Downloading BlazeFace detector model...")
        urllib.request.urlretrieve(DETECTOR_URL, DETECTOR_MODEL_PATH)
    if not os.path.exists(EMBEDDER_MODEL_PATH):
        raise FileNotFoundError(
            f"Place a MobileFaceNet ONNX model at {EMBEDDER_MODEL_PATH}. "
            "Export one from a pretrained MobileFaceNet checkpoint, or "
            "substitute any 112x112-in / 128-d-out face embedding ONNX model."
        )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectedFace:
    box: tuple            # (x, y, w, h) in pixels
    landmarks: list        # list of (x, y) keypoints, normalized
    embedding: Optional[np.ndarray] = None
    face_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class FaceDetectorService:
    def __init__(self, model_path: str = DETECTOR_MODEL_PATH):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)

    def detect(self, image_rgb: np.ndarray) -> list[DetectedFace]:
        """image_rgb: HxWx3 uint8 numpy array (RGB)."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._detector.detect(mp_image)

        h, w = image_rgb.shape[:2]
        faces = []
        for det in result.detections:
            bbox = det.bounding_box
            box = (bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
            landmarks = [(kp.x * w, kp.y * h) for kp in det.keypoints]
            faces.append(DetectedFace(box=box, landmarks=landmarks))
        return faces

    def close(self):
        self._detector.close()


# ---------------------------------------------------------------------------
# Alignment + cropping
# ---------------------------------------------------------------------------

def align_and_crop(image_rgb: np.ndarray, face: DetectedFace,
                    out_size: int = EMBEDDING_INPUT_SIZE) -> np.ndarray:
    """Rotate the image so the eyes are level, then crop+resize the face
    region to a square out_size x out_size image for the embedding model.
    MediaPipe's short-range detector returns 6 keypoints in this order:
    right_eye, left_eye, nose_tip, mouth_center, right_ear_tragion, left_ear_tragion.
    """
    img = Image.fromarray(image_rgb)

    if len(face.landmarks) >= 2:
        (rx, ry), (lx, ly) = face.landmarks[0], face.landmarks[1]
        angle = np.degrees(np.arctan2(ly - ry, lx - rx))
        img = img.rotate(angle, resample=Image.BICUBIC, center=(rx, ry))

    x, y, w, h = face.box
    pad = 0.25
    x0 = max(0, int(x - w * pad))
    y0 = max(0, int(y - h * pad))
    x1 = min(img.width, int(x + w * (1 + pad)))
    y1 = min(img.height, int(y + h * (1 + pad)))

    cropped = img.crop((x0, y0, x1, y1)).resize((out_size, out_size), Image.BICUBIC)
    return np.array(cropped)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class FaceEmbedder:
    def __init__(self, model_path: str = EMBEDDER_MODEL_PATH):
        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def embed(self, face_crop_rgb: np.ndarray) -> np.ndarray:
        """face_crop_rgb: 112x112x3 uint8 array -> returns L2-normalized 128-d vector."""
        img = face_crop_rgb.astype(np.float32)
        img = (img - 127.5) / 128.0                  # standard MobileFaceNet preprocessing
        img = np.transpose(img, (2, 0, 1))            # HWC -> CHW
        img = np.expand_dims(img, axis=0)              # add batch dim

        outputs = self._session.run(None, {self._input_name: img})
        embedding = outputs[0][0]
        return embedding / (np.linalg.norm(embedding) + 1e-10)


# ---------------------------------------------------------------------------
# Storage (SQLite for simplicity; swap for Postgres+pgvector at scale)
# ---------------------------------------------------------------------------

class FaceStore:
    def __init__(self, db_path: str = DB_PATH):
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                id TEXT PRIMARY KEY,
                name TEXT,
                centroid BLOB NOT NULL,
                face_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id TEXT PRIMARY KEY,
                photo_id TEXT NOT NULL,
                cluster_id TEXT,
                embedding BLOB NOT NULL,
                box TEXT,
                FOREIGN KEY (cluster_id) REFERENCES clusters(id)
            )
        """)
        self._conn.commit()

    # -- clusters --
    def get_all_clusters(self):
        rows = self._conn.execute("SELECT id, name, centroid, face_count FROM clusters").fetchall()
        return [
            {"id": r[0], "name": r[1], "centroid": np.frombuffer(r[2], dtype=np.float32), "face_count": r[3]}
            for r in rows
        ]

    def create_cluster(self, centroid: np.ndarray) -> str:
        cluster_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO clusters (id, name, centroid, face_count) VALUES (?, ?, ?, ?)",
            (cluster_id, None, centroid.astype(np.float32).tobytes(), 1),
        )
        self._conn.commit()
        return cluster_id

    def update_cluster_centroid(self, cluster_id: str, new_centroid: np.ndarray, face_count: int):
        self._conn.execute(
            "UPDATE clusters SET centroid = ?, face_count = ? WHERE id = ?",
            (new_centroid.astype(np.float32).tobytes(), face_count, cluster_id),
        )
        self._conn.commit()

    def label_cluster(self, cluster_id: str, name: str):
        self._conn.execute("UPDATE clusters SET name = ? WHERE id = ?", (name, cluster_id))
        self._conn.commit()

    # -- faces --
    def save_face(self, face_id: str, photo_id: str, embedding: np.ndarray, box: tuple, cluster_id: str):
        self._conn.execute(
            "INSERT INTO faces (id, photo_id, cluster_id, embedding, box) VALUES (?, ?, ?, ?, ?)",
            (face_id, photo_id, cluster_id, embedding.astype(np.float32).tobytes(), str(box)),
        )
        self._conn.commit()

    def get_all_faces(self):
        rows = self._conn.execute("SELECT id, photo_id, cluster_id, embedding FROM faces").fetchall()
        return [
            {"id": r[0], "photo_id": r[1], "cluster_id": r[2], "embedding": np.frombuffer(r[3], dtype=np.float32)}
            for r in rows
        ]

    def reassign_face_cluster(self, face_id: str, cluster_id: str):
        self._conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?", (cluster_id, face_id))
        self._conn.commit()

    def delete_cluster(self, cluster_id: str):
        self._conn.execute("DELETE FROM clusters WHERE id = ?", (cluster_id,))
        self._conn.commit()


# ---------------------------------------------------------------------------
# Clustering logic
# ---------------------------------------------------------------------------

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def assign_face_to_cluster(embedding: np.ndarray, store: FaceStore,
                            threshold: float = CLUSTER_THRESHOLD) -> str:
    """Incremental assignment: compare against existing cluster centroids,
    join the closest one within threshold, else create a new cluster."""
    clusters = store.get_all_clusters()

    best_id, best_dist = None, float("inf")
    for c in clusters:
        d = euclidean(embedding, c["centroid"])
        if d < best_dist:
            best_dist, best_id = d, c["id"]

    if best_id is not None and best_dist < threshold:
        cluster = next(c for c in clusters if c["id"] == best_id)
        n = cluster["face_count"] + 1
        # running mean update of the centroid
        new_centroid = cluster["centroid"] + (embedding - cluster["centroid"]) / n
        store.update_cluster_centroid(best_id, new_centroid, n)
        return best_id

    return store.create_cluster(embedding)


def batch_recluster(store: FaceStore, threshold: float = CLUSTER_THRESHOLD, min_samples: int = 2):
    """Periodic cleanup pass: re-runs DBSCAN over every face embedding to
    correct drift from incremental assignment. Rebuilds clusters, preserving
    labels where possible by majority vote of previous membership."""
    faces = store.get_all_faces()
    if len(faces) < min_samples:
        return

    embeddings = np.stack([f["embedding"] for f in faces])
    old_cluster_names = {c["id"]: c["name"] for c in store.get_all_clusters()}

    db = DBSCAN(eps=threshold, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(embeddings)

    # wipe and rebuild cluster table
    for c in store.get_all_clusters():
        store.delete_cluster(c["id"])

    label_to_new_id = {}
    for face, label in zip(faces, labels):
        if label == -1:
            # noise point: gets its own singleton cluster
            new_id = store.create_cluster(face["embedding"])
        else:
            if label not in label_to_new_id:
                new_id = store.create_cluster(face["embedding"])
                label_to_new_id[label] = new_id
                # try to preserve a name if the face's old cluster was named
                old_name = old_cluster_names.get(face["cluster_id"])
                if old_name:
                    store.label_cluster(new_id, old_name)
            else:
                new_id = label_to_new_id[label]
                cluster = next(c for c in store.get_all_clusters() if c["id"] == new_id)
                n = cluster["face_count"] + 1
                new_centroid = cluster["centroid"] + (face["embedding"] - cluster["centroid"]) / n
                store.update_cluster_centroid(new_id, new_centroid, n)

        store.reassign_face_cluster(face["id"], new_id)


# ---------------------------------------------------------------------------
# End-to-end pipeline for a single uploaded photo
# ---------------------------------------------------------------------------

class FacePipeline:
    def __init__(self):
        download_models()
        self.detector = FaceDetectorService()
        self.embedder = FaceEmbedder()
        self.store = FaceStore()

    def process_photo(self, image_bytes: bytes, photo_id: str) -> list[dict]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_rgb = np.array(image)

        faces = self.detector.detect(image_rgb)
        results = []
        for face in faces:
            crop = align_and_crop(image_rgb, face)
            embedding = self.embedder.embed(crop)
            cluster_id = assign_face_to_cluster(embedding, self.store)
            self.store.save_face(face.face_id, photo_id, embedding, face.box, cluster_id)
            results.append({
                "face_id": face.face_id,
                "box": face.box,
                "cluster_id": cluster_id,
            })
        return results

    def close(self):
        self.detector.close()
