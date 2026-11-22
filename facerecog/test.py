from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=-1)

faces = app.get("test.jpg")

print(f"Faces detected: {len(faces)}")

for i, face in enumerate(faces):
    print(f"Face {i}")
    print("Bounding box:", face.bbox)
    print("Embedding dimensions:", len(face.embedding))