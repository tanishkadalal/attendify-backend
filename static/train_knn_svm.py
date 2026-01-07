import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

X = np.load("../svm_data/X_embeddings.npy")
y = np.load("../svm_data/y_labels.npy")

knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
knn.fit(X, y)

svm = SVC(kernel="linear", probability=True)
svm.fit(X, y)

with open("../svm_data/knn.pkl", "wb") as f:
    pickle.dump(knn, f)

with open("../svm_data/svm.pkl", "wb") as f:
    pickle.dump(svm, f)

print("✅ KNN & SVM trained")
