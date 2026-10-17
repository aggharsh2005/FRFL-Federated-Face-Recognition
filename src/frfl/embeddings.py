"""InsightFace embedding extraction and cached dataset construction."""
import glob
import os
import cv2
import numpy as np

def extract_embedding(image_path, face_app):
    """Extract the largest detected face's ArcFace embedding, or None."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = face_app.get(img)
    if not faces:
        return None
    faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].embedding

def build_or_load_embedding_dataset(dataset_dir, embedding_dir, class_names, label_encoder, face_app):
    """Load cached arrays or build embeddings from class-named image folders."""
    os.makedirs(embedding_dir, exist_ok=True)
    x_path = os.path.join(embedding_dir, "X.npy")
    y_path = os.path.join(embedding_dir, "y.npy")
    names_path = os.path.join(embedding_dir, "label_names.npy")
    if all(os.path.exists(p) for p in (x_path, y_path, names_path)):
        X = np.load(x_path)
        y = np.load(y_path)
        label_names = np.load(names_path, allow_pickle=True)
        print("Loaded existing embeddings from disk.")
    else:
        X_list, y_list = [], []
        for person in class_names:
            person_dir = os.path.join(dataset_dir, person)
            for img_path in glob.glob(os.path.join(person_dir, "*")):
                embedding = extract_embedding(img_path, face_app)
                if embedding is not None:
                    X_list.append(embedding)
                    y_list.append(person)
        X = np.asarray(X_list, dtype=np.float32)
        y = label_encoder.transform(y_list)
        label_names = label_encoder.classes_
        np.save(x_path, X)
        np.save(y_path, y)
        np.save(names_path, label_names)
        print("Generated and saved new embeddings.")
    print("Number of embeddings:", X.shape[0])
    print("Embedding dimension:", X.shape[1] if len(X.shape) > 1 else 0)
    return X, y, label_names
