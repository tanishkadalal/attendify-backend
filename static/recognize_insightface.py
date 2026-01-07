import cv2
import pickle
import numpy as np
from datetime import datetime, timedelta
import os

# ======================================================
# PATH SETUP
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMB_PATH = os.path.join(BASE_DIR, "insight_embeddings.pkl")
SERVICE_KEY = os.path.join(BASE_DIR, "serviceAccountKey.json")
STATUS_FILE = os.path.join(BASE_DIR, "status.txt")

# ======================================================
# FIREBASE
# ======================================================
import firebase_admin
from firebase_admin import credentials, firestore

# ======================================================
# INSIGHTFACE
# ======================================================
from insightface.app import FaceAnalysis

# ======================================================
# CONFIG
# ======================================================
THRESHOLD = 0.45
ATTENDANCE_COOLDOWN_MINUTES = 60

# ======================================================
# GLOBAL STATS
# ======================================================
total_faces = 0
recognized_faces = 0

# ======================================================
# FIREBASE INIT (SAFE)
# ======================================================
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_KEY)
    firebase_admin.initialize_app(cred)

firestore_db = firestore.client()

# ======================================================
# LOAD EMBEDDINGS
# ======================================================
with open(EMB_PATH, "rb") as f:
    emb_db = pickle.load(f)

db_embeddings = []
db_labels = []

for person_name, embs in emb_db.items():
    for emb in embs:
        db_embeddings.append(emb)
        db_labels.append(person_name)

db_embeddings = np.vstack(db_embeddings)

# ======================================================
# INSIGHTFACE MODEL
# ======================================================
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

# ======================================================
# ATTENDANCE FUNCTION
# ======================================================
def mark_attendance(label, confidence):
    try:
        name, roll_no = label.rsplit(" ", 1)
        now = datetime.utcnow()

        records = (
            firestore_db.collection("attendance")
            .where("student_id", "==", roll_no)
            .stream()
        )

        for record in records:
            last_time = record.to_dict().get("created_at")
            if last_time:
                last_time = last_time.replace(tzinfo=None)
                if now - last_time < timedelta(minutes=ATTENDANCE_COOLDOWN_MINUTES):
                    return

        firestore_db.collection("attendance").add({
            "student_id": roll_no,
            "name": name,
            "status": "Present",
            "confidence": float(confidence),
            "marked_by": "face_recognition",
            "date": now.date().isoformat(),
            "time": now.strftime("%H:%M:%S"),
            "created_at": firestore.SERVER_TIMESTAMP
        })

    except Exception as e:
        print("Attendance error:", e)

# ======================================================
# CAMERA + RECOGNITION LOOP (BACKGROUND)
# ======================================================
def run_camera_recognition():
    global total_faces, recognized_faces

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        with open(STATUS_FILE, "w") as f:
            f.write("Camera not accessible")
        return

    with open(STATUS_FILE, "w") as f:
        f.write("Camera started. Waiting for face...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)

        if not faces:
            continue

        for face in faces:
            total_faces += 1

            emb = face.embedding
            emb = emb / np.linalg.norm(emb)

            sims = np.dot(db_embeddings, emb)
            best_idx = np.argmax(sims)
            best_score = sims[best_idx]

            if best_score > THRESHOLD:
                label = db_labels[best_idx]
                recognized_faces += 1
                mark_attendance(label, best_score)

                status_text = f"Detected: {label} ({best_score:.2f})"
            else:
                status_text = "Unknown face detected"

            # ✅ WRITE STATUS FOR FRONTEND
            with open(STATUS_FILE, "w") as f:
                f.write(status_text)

            print(status_text)

    cap.release()

    # FINAL STATS
    accuracy = (recognized_faces / total_faces) * 100 if total_faces else 0.0
    print("\nFINAL STATS")
    print("Total faces:", total_faces)
    print("Recognized:", recognized_faces)
    print("Accuracy:", accuracy)

# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    run_camera_recognition()
