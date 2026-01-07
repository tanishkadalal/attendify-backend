import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import random

INPUT_DIR = "../Data"
OUTPUT_DIR = "../Data_processed"
IMG_SIZE = 224
AUG_PER_IMAGE = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

mtcnn = MTCNN(
    image_size=IMG_SIZE,
    margin=40,
    keep_all=False,
    min_face_size=30,
    thresholds=[0.6, 0.7, 0.7]
)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def augment(img):
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.5:
        v = random.randint(-20, 20)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 2] = np.clip(hsv[..., 2] + v, 0, 255)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

for person in os.listdir(INPUT_DIR):
    in_p = os.path.join(INPUT_DIR, person)
    out_p = os.path.join(OUTPUT_DIR, person)

    if not os.path.isdir(in_p):
        continue

    os.makedirs(out_p, exist_ok=True)
    idx = 0
    print(f"➡️ Processing folder: {person}")

    for img_name in os.listdir(in_p):
        img_path = os.path.join(in_p, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(rgb)

        if face is None:
            print(f"[SKIP] No face detected in {img_name}")
            continue

        # 🔥 FIX 1: scale properly
        face = face.permute(1, 2, 0).numpy()
        face = (face * 255).astype(np.uint8)

        face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

        # 🔥 FIX 2: CLAHE ONLY on L channel
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        face = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        for _ in range(AUG_PER_IMAGE):
            aug_face = augment(face)
            cv2.imwrite(f"{out_p}/{idx}.jpg", aug_face)
            idx += 1

print("✅ Preprocessing done → Data_processed/")
