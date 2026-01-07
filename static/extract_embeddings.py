import torch
import torch.nn as nn
from torchvision import models, transforms
from facenet_pytorch import MTCNN
import numpy as np
import os
import cv2

DATA_DIR = "../Data_processed"
SAVE_DIR = "../svm_data"
EMB_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet.fc = nn.Identity()
resnet.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

X, y, labels = [], [], []

label_map = {name: i for i, name in enumerate(os.listdir(DATA_DIR))}

for person in os.listdir(DATA_DIR):
    p_dir = os.path.join(DATA_DIR, person)
    if not os.path.isdir(p_dir):
        continue

    for img_name in os.listdir(p_dir):
        img = cv2.imread(os.path.join(p_dir, img_name))
        if img is None:
            continue

        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = resnet(tensor)
            emb = emb / emb.norm(dim=1, keepdim=True)

        X.append(emb.cpu().numpy()[0])
        y.append(label_map[person])
        print("Total embeddings collected:", len(X))


labels = list(label_map.keys())

np.save(f"{SAVE_DIR}/X_embeddings.npy", np.array(X))
np.save(f"{SAVE_DIR}/y_labels.npy", np.array(y))
np.save(f"{SAVE_DIR}/labels.npy", np.array(labels))
print("Total embeddings collected:", len(X))

print("✅ Embeddings saved")
