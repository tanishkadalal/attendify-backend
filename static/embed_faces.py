import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

DATA_DIR = "../Data"
SAVE_PATH = "../insight_embeddings.pkl"

app = FaceAnalysis(
    name="buffalo_l",  # ArcFace 512D
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

embeddings_db = {}

for person in os.listdir(DATA_DIR):
    person_path = os.path.join(DATA_DIR, person)
    if not os.path.isdir(person_path):
        continue

    person_embeddings = []

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        faces = app.get(img)
        if len(faces) == 0:
            continue

        # choose largest face
        face = max(
            faces,
            key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])
        )

        emb = face.embedding
        emb = emb / np.linalg.norm(emb)
        person_embeddings.append(emb)

    if len(person_embeddings) > 0:
        embeddings_db[person] = np.vstack(person_embeddings)

    print(f"{person}: {len(person_embeddings)} embeddings")

with open(SAVE_PATH, "wb") as f:
    pickle.dump(embeddings_db, f)

print("✅ Embeddings saved to insight_embeddings.pkl")
