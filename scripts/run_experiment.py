"""FRFL end-to-end workflow extracted from the companion notebook.

Run in Google Colab after installing requirements-colab.txt. This script keeps
all notebook code cells in order and preserves interactive Colab file uploads.
"""


# ========================================================================
# Original notebook cell 2: Step 0 — Install Dependencies
# ========================================================================
# Install before running: pip install -r requirements-colab.txt
print("Install requirements-colab.txt in Colab before running the experiment.")


# ========================================================================
# Original notebook cell 3: Step 0 — Install Dependencies
# ========================================================================
# ADDED: core imports used throughout the notebook
import os
import random
import cv2
import copy
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder

import onnxruntime as ort
from insightface.app import FaceAnalysis
from google.colab import files
from IPython.display import display  # ADDED: for rendering before/after comparison tables

print("All libraries imported successfully.")


# ========================================================================
# Original notebook cell 5: Step 1 — Mount Google Drive and Create the FRFL Project Folder
# ========================================================================
# ADDED: mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# ADDED: new project folder for this experiment
BASE_DIR = "/content/drive/MyDrive/FRFL"

# ADDED: required sub-folders
# - updated_samples_A : new MASK images of the identity being updated (used by Model_A)
# - updated_samples_B : new SUNGLASSES+DOG images of the identity being updated (used by Model_B)
# - updated_samples_C : new GLASSES images of the identity being updated (used by Model_C)
FOLDERS = [
    "dataset",
    "embeddings",
    "local_models",
    "federated_model",
    "evaluation",
    "results",
    "updated_samples_A",
    "updated_samples_B",
    "updated_samples_C",
    "test_images",
]

os.makedirs(BASE_DIR, exist_ok=True)
for folder in FOLDERS:
    path = os.path.join(BASE_DIR, folder)
    os.makedirs(path, exist_ok=True)
    print("Ready:", path)


# ========================================================================
# Original notebook cell 7: Step 2 — GPU / Execution Provider Check
# ========================================================================
# ADDED: check available ONNX Runtime providers
available_providers = ort.get_available_providers()
print("Available ONNX Runtime providers:", available_providers)

if "CUDAExecutionProvider" in available_providers:
    ACTIVE_PROVIDER = "CUDAExecutionProvider"
    CTX_ID = 0  # GPU device id
else:
    ACTIVE_PROVIDER = "CPUExecutionProvider"
    CTX_ID = -1  # CPU

print("Active provider selected:", ACTIVE_PROVIDER)


# ========================================================================
# Original notebook cell 9: Step 3 — Dataset Discovery
# ========================================================================
# ADDED: dynamically discover identity classes from the dataset folder
DATASET_DIR = "/content/drive/MyDrive/FRFL/dataset/105_classes_pins_dataset"

class_names = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

NUM_CLASSES = len(class_names)

label_encoder = LabelEncoder()
label_encoder.fit(class_names)

# ADDED: single source of truth for the identity being updated across the whole notebook
TARGET_IDENTITY = "Sunil Grover"

print("Discovered classes:", class_names)
print("Number of classes:", NUM_CLASSES)

if TARGET_IDENTITY not in class_names:
    print(f"\nWARNING: '{TARGET_IDENTITY}' folder not found in dataset/. "
          f"Please make sure FRFL/dataset/{TARGET_IDENTITY}/ exists with plain images.")


# ========================================================================
# Original notebook cell 11: Step 4 — Load InsightFace (`buffalo_l`)
# ========================================================================
# ADDED: load InsightFace buffalo_l model (RetinaFace + ArcFace)
face_app = FaceAnalysis(name="buffalo_l", providers=[ACTIVE_PROVIDER])
face_app.prepare(ctx_id=CTX_ID, det_size=(640, 640))

print("Number of classes:", NUM_CLASSES)
print("Provider:", ACTIVE_PROVIDER)
print("Model loaded successfully.")


# ========================================================================
# Original notebook cell 13: Step 5 — ArcFace Embedding Extraction
# ========================================================================
# ADDED: helper function -> extract a single ArcFace embedding from an image path
def extract_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    faces = face_app.get(img)
    if len(faces) == 0:
        return None
    faces = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    return faces[0].embedding  # 512-D ArcFace embedding


# ========================================================================
# Original notebook cell 14: Step 5 — ArcFace Embedding Extraction
# ========================================================================
from google.colab import drive
drive.mount('/content/drive')


# ========================================================================
# Original notebook cell 15: Step 5 — ArcFace Embedding Extraction
# ========================================================================
# ADDED: build (or load) the full embedding dataset
EMBED_DIR = os.path.join(BASE_DIR, "embeddings")
X_PATH = os.path.join(EMBED_DIR, "X.npy")
Y_PATH = os.path.join(EMBED_DIR, "y.npy")
NAMES_PATH = os.path.join(EMBED_DIR, "label_names.npy")

if os.path.exists(X_PATH) and os.path.exists(Y_PATH) and os.path.exists(NAMES_PATH):
    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    label_names = np.load(NAMES_PATH, allow_pickle=True)
    print("Loaded existing embeddings from disk.")
else:
    X_list, y_list = [], []
    for person in class_names:
        person_dir = os.path.join(DATASET_DIR, person)
        image_paths = glob.glob(os.path.join(person_dir, "*"))
        for img_path in image_paths:
            emb = extract_embedding(img_path)
            if emb is not None:
                X_list.append(emb)
                y_list.append(person)

    X = np.array(X_list, dtype=np.float32)
    y = label_encoder.transform(y_list)
    label_names = label_encoder.classes_

    np.save(X_PATH, X)
    np.save(Y_PATH, y)
    np.save(NAMES_PATH, label_names)
    print("Generated and saved new embeddings.")

print("Number of embeddings:", X.shape[0])
print("Embedding dimension:", X.shape[1] if len(X.shape) > 1 else 0)


# ========================================================================
# Original notebook cell 17: Step 6 — MLP Classifier and Shared Training Settings
# ========================================================================
# ADDED: simple MLP classifier on top of ArcFace embeddings
class FaceMLP(nn.Module):
    def __init__(self, num_classes, input_dim=512):
        super(FaceMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch device:", DEVICE)

# ADDED: shared training settings used by ALL models
TRAIN_EPOCHS = 30   # baseline training -- 30 gives good convergence
RETRAIN_EPOCHS = 30   # FedProx retraining (Model_A / Model_B updates)
LEARNING_RATE = 1e-3
BATCH_SIZE = 16

# ADDED: FedProx proximal coefficient -- higher mu = local update stays closer to the
# reference weights = more gradual/controlled change.
FEDPROX_MU = 0.05

# ADDED: fixed seed + reproducibility helper. WHY THIS MATTERS: Model_A, Model_B, Model_C
# all start from the exact same Round-0 weights and train on the exact same (X, y) data --
# but DataLoader(shuffle=True) and nn.Dropout both draw from torch's global RNG. Without a
# fixed seed, each model's training loop shuffles batches differently and drops out different
# neurons each run, so the three "identical" trainings quietly drift to different local minima
# -- which is why Step 9's baseline predictions could disagree even though nothing about the
# model/data was actually different. Calling set_seed(SEED) with the SAME seed right before
# each model's baseline training resets that RNG state every time, so all three see the exact
# same shuffle order and dropout pattern and converge identically.
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========================================================================
# Original notebook cell 19: Step 7 — Training, FedProx, Diff-Weighted Aggregation, and Inference Helpers
# ========================================================================
# ADDED: build a DataLoader from embeddings + labels
def make_dataloader(X_data, y_data, batch_size=BATCH_SIZE, shuffle=True):
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ========================================================================
# Original notebook cell 20: Step 7 — Training, FedProx, Diff-Weighted Aggregation, and Inference Helpers
# ========================================================================
# ADDED: plain training function -- used for the baseline training of Model_A/B/C
def train_model(model, X_data, y_data, epochs, lr=LEARNING_RATE):
    model = model.to(DEVICE)
    model.train()

    dataloader = make_dataloader(X_data, y_data)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

    return model, loss_history


# ========================================================================
# Original notebook cell 21: Step 7 — Training, FedProx, Diff-Weighted Aggregation, and Inference Helpers
# ========================================================================
# ADDED: FedProx training function -- adds a proximal term (mu/2) * ||w - w_ref||^2
# that keeps the local update from drifting too far, too fast, from a reference model.
def train_model_fedprox(model, X_data, y_data, global_ref_state_dict, epochs, lr=LEARNING_RATE, mu=0.01):
    model = model.to(DEVICE)
    model.train()

    global_ref_params = {
        name: param.detach().clone().to(DEVICE)
        for name, param in global_ref_state_dict.items()
    }

    dataloader = make_dataloader(X_data, y_data)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            ce_loss = criterion(outputs, batch_y)

            proximal_term = 0.0
            for name, param in model.named_parameters():
                proximal_term += torch.sum((param - global_ref_params[name]) ** 2)

            loss = ce_loss + (mu / 2) * proximal_term
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs} - loss (CE + proximal): {avg_loss:.4f}")

    return model, loss_history


# ========================================================================
# Original notebook cell 22: Step 7 — Training, FedProx, Diff-Weighted Aggregation, and Inference Helpers
# ========================================================================

# ADDED: measure how much a model's weights changed relative to a FIXED reference
# (the Round-0 shared initial weights, used consistently everywhere in this notebook so
# every drift number is directly comparable -- no self-reference issues).
def compute_weight_diff(state_dict, reference_state_dict):
    total_diff = 0.0
    for key in state_dict.keys():
        current_tensor = state_dict[key].float()
        # ADDED: move the reference tensor to the SAME device as the current tensor first
        # (models get moved to GPU during training, but saved reference snapshots can end
        # up on CPU -- this avoids "Expected all tensors to be on the same device" errors).
        reference_tensor = reference_state_dict[key].float().to(current_tensor.device)
        diff = current_tensor - reference_tensor
        total_diff += torch.sum(diff ** 2).item()
    return total_diff ** 0.5


# ========================================================================
# Original notebook cell 23: Step 7 — Training, FedProx, Diff-Weighted Aggregation, and Inference Helpers
# ========================================================================
# ADDED: DIFF-WEIGHTED aggregation. Each participant's contribution is weighted by how
# much it changed relative to 'reference_state_dict' (always the Round-0 shared initial
# weights in this notebook). Bigger drift -> bigger coefficient. epsilon avoids
# divide-by-zero / total exclusion when a participant's drift is exactly zero.
def weighted_federated_average(state_dicts, names, reference_state_dict, epsilon=1e-6):
    diffs = [compute_weight_diff(sd, reference_state_dict) for sd in state_dicts]
    adjusted = [d + epsilon for d in diffs]
    total = sum(adjusted)
    coefficients = [a / total for a in adjusted]

    print("Diff-weighted aggregation coefficients:")
    for name, diff, coef in zip(names, diffs, coefficients):
        print(f"  {name}: weight difference = {diff:.6f}  ->  aggregation coefficient = {coef:.4f}")

    # ADDED: pick one common device (CPU) up front, and move every tensor there before
    # combining -- avoids "Expected all tensors to be on the same device" errors when some
    # models were trained on GPU and others (like a freshly loaded Global model) are on CPU.
    target_device = torch.device("cpu")

    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state.keys():
        avg_state[key] = coefficients[0] * state_dicts[0][key].float().to(target_device)
        for i in range(1, len(state_dicts)):
            term = coefficients[i] * state_dicts[i][key].float().to(target_device)
            avg_state[key] = avg_state[key] + term

    return avg_state, coefficients


# ========================================================================
# Original notebook cell 24: Step 7 — Training, FedProx, Diff-Weighted Aggregation, and Inference Helpers
# ========================================================================
# ADDED: top-k inference helper. Returns [(class_name, probability), ...] sorted from
# most to least confident.
def predict_topk(image_path, model, label_encoder, k=2):
    img = cv2.imread(image_path)
    if img is None:
        print("Could not read image:", image_path)
        return None

    faces = face_app.get(img)
    if len(faces) == 0:
        print("No face detected in the image.")
        return None

    faces = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    embedding = faces[0].embedding

    model = model.to(DEVICE)
    model.eval()
    x_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_indices = np.argsort(probs)[::-1][:k]
    results = []
    for idx in top_indices:
        name = label_encoder.inverse_transform([idx])[0]
        results.append((name, float(probs[idx])))
    return results


# ========================================================================
# Original notebook cell 25: Step 7 — Training, FedProx, Diff-Weighted Aggregation, and Inference Helpers
# ========================================================================
# ADDED: shared BEFORE-vs-AFTER helper -- used for Model_A, Model_B, and Model_C.
# Builds a small pandas table (BEFORE row = baseline model, AFTER row = updated model) on the
# same appearance-changed test image, displays it, and prints a short analysis of what the
# retraining actually changed (flip to correct identity / confidence delta / regression check).
def before_after_analysis(model_name, baseline_model, updated_model, test_image_path,
                           appearance_label, label_encoder, target_identity=TARGET_IDENTITY):
    before_result = predict_topk(test_image_path, baseline_model, label_encoder, k=2)
    after_result = predict_topk(test_image_path, updated_model, label_encoder, k=2)

    rows = []
    if before_result is not None:
        (top_c, top_p), (sec_c, sec_p) = before_result
        rows.append({
            "Stage": "BEFORE",
            "Predicted": top_c,
            "Confidence (%)": round(top_p * 100, 2),
            "2nd Best": sec_c,
            "2nd Confidence (%)": round(sec_p * 100, 2),
        })
    if after_result is not None:
        (top_c, top_p), (sec_c, sec_p) = after_result
        rows.append({
            "Stage": "AFTER",
            "Predicted": top_c,
            "Confidence (%)": round(top_p * 100, 2),
            "2nd Best": sec_c,
            "2nd Confidence (%)": round(sec_p * 100, 2),
        })

    df = pd.DataFrame(rows)
    print(f"=== {model_name} on {appearance_label} Image: BEFORE vs AFTER ===")
    display(df)

    if before_result is not None and after_result is not None:
        before_top_c, before_top_p = before_result[0]
        after_top_c, after_top_p = after_result[0]
        before_correct = (before_top_c == target_identity)
        after_correct = (after_top_c == target_identity)

        print("\nAnalysis:")
        if after_correct and not before_correct:
            print(f"  -> {model_name} FLIPPED to the correct identity ('{target_identity}') "
                  f"after learning the {appearance_label} images.")
        elif after_correct and before_correct:
            delta = (after_top_p - before_top_p) * 100
            direction = "increased" if delta >= 0 else "decreased"
            print(f"  -> {model_name} was already correct; confidence {direction} by "
                  f"{abs(delta):.2f} points after the update.")
        elif not after_correct and before_correct:
            print(f"  -> WARNING: {model_name} was correct BEFORE but is WRONG after the "
                  f"update (possible regression -- check retrain data / mu).")
        else:
            print(f"  -> {model_name} still does not predict '{target_identity}' as top-1 "
                  f"after the update.")

    return df


# ========================================================================
# Original notebook cell 27: Step 8 — Initialize the Round-0 Global Model, then Train Model_A, Model_B, Model_C
# ========================================================================
# ADDED: create the Round-0 Global Model (fresh random init).
global_model = FaceMLP(num_classes=NUM_CLASSES)
ROUND0_REFERENCE = copy.deepcopy(global_model.state_dict())  # fixed reference for ALL diffs below

ROUND0_PATH = os.path.join(BASE_DIR, "federated_model", "global_model_round0.pth")
torch.save(ROUND0_REFERENCE, ROUND0_PATH)
print("Saved Round-0 Global Model to:", ROUND0_PATH)


# ========================================================================
# Original notebook cell 28: Step 8 — Initialize the Round-0 Global Model, then Train Model_A, Model_B, Model_C
# ========================================================================
# ADDED: train Model_A, Model_B, Model_C independently on the SAME base embeddings
# (X, y), all starting from the SAME Round-0 Global Model.
LOCAL_MODELS_DIR = os.path.join(BASE_DIR, "local_models")

model_names = ["Model_A", "Model_B", "Model_C"]
models = {}
loss_histories = {}

for name in model_names:
    print(f"\nTraining {name} (baseline) ...")
    # FIXED: reset the RNG to the SAME seed before every model so shuffle order + dropout
    # pattern match across Model_A/B/C -- they truly train identically at this stage.
    set_seed(SEED)
    model = FaceMLP(num_classes=NUM_CLASSES)
    model.load_state_dict(copy.deepcopy(ROUND0_REFERENCE))
    trained_model, loss_hist = train_model(model, X, y, epochs=TRAIN_EPOCHS)

    models[name] = trained_model
    loss_histories[name] = loss_hist

    save_path = os.path.join(LOCAL_MODELS_DIR, f"{name}_baseline.pth")
    torch.save(trained_model.state_dict(), save_path)
    print(f"Saved: {save_path}")

print("\nAll three models trained independently on the base dataset (no appearance updates yet).")


# ========================================================================
# Original notebook cell 30: Step 9 — Baseline Prediction Check (Plain Sunil Grover Image)
# ========================================================================
# ADDED: upload a plain (no mask, no sunglasses) test image of Sunil Grover
print("Please upload a PLAIN test image of Sunil Grover (no mask, no sunglasses)...")
uploaded_plain = files.upload()
plain_test_image_path = list(uploaded_plain.keys())[0]
print("Using plain test image:", plain_test_image_path)


# ========================================================================
# Original notebook cell 31: Step 9 — Baseline Prediction Check (Plain Sunil Grover Image)
# ========================================================================
# ADDED: baseline sanity check -- predict with all three models on the plain image
print("=== Baseline Predictions (Plain Sunil Grover Image) ===")
for name in model_names:
    result = predict_topk(plain_test_image_path, models[name], label_encoder, k=2)
    if result is not None:
        (top_class, top_prob), (second_class, second_prob) = result
        print(f"{name}: Predicted = {top_class} ({top_prob * 100:.2f}%), "
              f"Second Best = {second_class} ({second_prob * 100:.2f}%)")


# ========================================================================
# Original notebook cell 33: Step 10 — Build the Global Model from the Baseline Consensus
# ========================================================================
# ADDED: Global Model = diff-weighted average of the three baseline-trained models
baseline_state_dicts = [
    copy.deepcopy(models["Model_A"].state_dict()),
    copy.deepcopy(models["Model_B"].state_dict()),
    copy.deepcopy(models["Model_C"].state_dict()),
]

# Ensure ROUND0_REFERENCE is on the same device as other state_dicts
ROUND0_REFERENCE_ON_DEVICE = {k: v.to(DEVICE) for k, v in ROUND0_REFERENCE.items()}

global_state_dict, baseline_coefficients = weighted_federated_average(
    baseline_state_dicts,
    names=["Model_A", "Model_B", "Model_C"],
    reference_state_dict=ROUND0_REFERENCE_ON_DEVICE,
)

global_model = FaceMLP(num_classes=NUM_CLASSES)
global_model.load_state_dict(global_state_dict)

GLOBAL_BASELINE_PATH = os.path.join(BASE_DIR, "federated_model", "global_model_baseline.pth")
torch.save(global_model.state_dict(), GLOBAL_BASELINE_PATH)
print("Saved baseline Global Model to:", GLOBAL_BASELINE_PATH)

# ADDED: fixed snapshot of the baseline Global Model, used as the FedProx PROXIMAL reference
# for Model_A's and Model_B's retraining below. This is DIFFERENT from ROUND0_REFERENCE:
# - ROUND0_REFERENCE (random init) is only used for DIFF-WEIGHTING (measuring how much a
#   model changed, for aggregation coefficients) -- that's just a measurement, not a loss term.
# - BASELINE_GLOBAL_REFERENCE (this one) is used inside the FedProx LOSS during retraining.
#   It must be a model that already has real learned knowledge (not random noise), otherwise
#   the proximal term would fight against everything Model_A/Model_B already learned in
#   Step 8's baseline training, hurting convergence.
BASELINE_GLOBAL_REFERENCE = copy.deepcopy(global_model.state_dict())


# ========================================================================
# Original notebook cell 35: Step 11 — Update Model_A with Mask Images of Sunil Grover
# ========================================================================
# ADDED: load and embed the new MASK images (same unchanged ArcFace pipeline)
UPDATED_DIR_A = os.path.join(BASE_DIR, "updated_samples_A")

new_X_A, new_y_A = [], []
if os.path.isdir(UPDATED_DIR_A):
    updated_people_A = [
        d for d in os.listdir(UPDATED_DIR_A)
        if os.path.isdir(os.path.join(UPDATED_DIR_A, d))
    ]

    # FIXED: match folder names to label_encoder classes CASE-INSENSITIVELY. Drive folder
    # names (e.g. "sunil grover") often don't match the exact casing label_encoder was fit
    # on (e.g. "Sunil Grover"), which used to crash deep inside sklearn with a cryptic
    # "unseen labels" KeyError. We build a lowercase -> correct-cased lookup once, and use
    # the CORRECTLY CASED label for training while still reading images from the folder's
    # actual (whatever-cased) path on disk.
    class_lookup_A = {c.lower(): c for c in label_encoder.classes_}

    unmatched_A = [p for p in updated_people_A if p.lower() not in class_lookup_A]
    if unmatched_A:
        raise ValueError(
            f"Folder name(s) {unmatched_A} under 'updated_samples_A/' don't match "
            f"any class label_encoder was trained on, even case-insensitively. Check for typos "
            f"or extra spaces. (TARGET_IDENTITY is currently '{TARGET_IDENTITY}'.)"
        )

    for person in updated_people_A:
        correct_label = class_lookup_A[person.lower()]  # ADDED: correctly-cased label
        person_dir = os.path.join(UPDATED_DIR_A, person)
        image_paths = glob.glob(os.path.join(person_dir, "*"))
        for img_path in image_paths:
            emb = extract_embedding(img_path)
            if emb is not None:
                new_X_A.append(emb)
                new_y_A.append(correct_label)  # FIXED: use correct-cased label, not raw folder name

if len(new_X_A) > 0:
    new_X_A = np.array(new_X_A, dtype=np.float32)
    new_y_A = label_encoder.transform(new_y_A)
    X_updated_A = np.vstack([X, new_X_A])
    y_updated_A = np.concatenate([y, new_y_A])
    print(f"Found {len(new_X_A)} new mask embeddings. Updated Model_A training set size: {X_updated_A.shape[0]}")
else:
    X_updated_A = X.copy()
    y_updated_A = y.copy()
    print(f"WARNING: no images found in 'updated_samples_A/'. Add "
          f"'updated_samples_A/{TARGET_IDENTITY}/*.jpg' (mask photos) before continuing.")


# ========================================================================
# Original notebook cell 36: Step 11 — Update Model_A with Mask Images of Sunil Grover
# ========================================================================
# CHANGED: retrain ONLY Model_A on the updated (base + mask) data -- plain MLP retrain,
# NO FedProx here. Individual city updates should learn freely from their own combined
# old + new samples; FedProx is only applied later (Step 15.5) when the cities converge
# around the shared Global Model.
# Model_B and Model_C are NOT touched.
print("Retraining Model_A (plain MLP retrain, no FedProx) on updated (base + mask) data ...")
models["Model_A"], retrain_loss_A = train_model(
    models["Model_A"],
    X_updated_A,   # combined old (base) + new (mask) samples
    y_updated_A,
    epochs=RETRAIN_EPOCHS,
)

save_path_A = os.path.join(LOCAL_MODELS_DIR, "Model_A_updated.pth")
torch.save(models["Model_A"].state_dict(), save_path_A)
print("Saved updated Model_A to:", save_path_A)
print("\nModel_B and Model_C were NOT touched.")


# ========================================================================
# Original notebook cell 38: Step 11.5 — Before vs After: Model_A on the Mask Image
# ========================================================================
# ADDED: upload a MASK test image of Sunil Grover (used for the before-vs-after check)
print("Please upload a MASK test image of Sunil Grover...")
uploaded_mask_check = files.upload()
mask_check_image_path = list(uploaded_mask_check.keys())[0]
print("Using mask test image:", mask_check_image_path)


# ========================================================================
# Original notebook cell 39: Step 11.5 — Before vs After: Model_A on the Mask Image
# ========================================================================
# ADDED: BEFORE vs AFTER comparison -- Model_A on the mask image, using the shared
# before_after_analysis() helper (table + analysis) instead of plain prints.
# BEFORE = baseline Model_A (saved in Step 8, prior to the Step 11 mask retraining)
# AFTER  = Model_A after Step 11's FedProx retraining on (base + mask) data

baseline_model_A = FaceMLP(num_classes=NUM_CLASSES)
baseline_model_A.load_state_dict(
    torch.load(os.path.join(LOCAL_MODELS_DIR, "Model_A_baseline.pth"), map_location=DEVICE)
)

df_before_after_A = before_after_analysis(
    "Model_A", baseline_model_A, models["Model_A"], mask_check_image_path, "Mask", label_encoder
)


# ========================================================================
# Original notebook cell 41: Step 12 — Update Model_B with Sunglasses + Dog Images of Sunil Grover
# ========================================================================
# ADDED: load and embed the new SUNGLASSES+DOG images (same unchanged ArcFace pipeline)
UPDATED_DIR_B = os.path.join(BASE_DIR, "updated_samples_B")

new_X_B, new_y_B = [], []
if os.path.isdir(UPDATED_DIR_B):
    updated_people_B = [
        d for d in os.listdir(UPDATED_DIR_B)
        if os.path.isdir(os.path.join(UPDATED_DIR_B, d))
    ]

    # FIXED: match folder names to label_encoder classes CASE-INSENSITIVELY. Drive folder
    # names (e.g. "sunil grover") often don't match the exact casing label_encoder was fit
    # on (e.g. "Sunil Grover"), which used to crash deep inside sklearn with a cryptic
    # "unseen labels" KeyError. We build a lowercase -> correct-cased lookup once, and use
    # the CORRECTLY CASED label for training while still reading images from the folder's
    # actual (whatever-cased) path on disk.
    class_lookup_B = {c.lower(): c for c in label_encoder.classes_}

    unmatched_B = [p for p in updated_people_B if p.lower() not in class_lookup_B]
    if unmatched_B:
        raise ValueError(
            f"Folder name(s) {unmatched_B} under 'updated_samples_B/' don't match "
            f"any class label_encoder was trained on, even case-insensitively. Check for typos "
            f"or extra spaces. (TARGET_IDENTITY is currently '{TARGET_IDENTITY}'.)"
        )

    for person in updated_people_B:
        correct_label = class_lookup_B[person.lower()]  # ADDED: correctly-cased label
        person_dir = os.path.join(UPDATED_DIR_B, person)
        image_paths = glob.glob(os.path.join(person_dir, "*"))
        for img_path in image_paths:
            emb = extract_embedding(img_path)
            if emb is not None:
                new_X_B.append(emb)
                new_y_B.append(correct_label)  # FIXED: use correct-cased label, not raw folder name

if len(new_X_B) > 0:
    new_X_B = np.array(new_X_B, dtype=np.float32)
    new_y_B = label_encoder.transform(new_y_B)
    X_updated_B = np.vstack([X, new_X_B])
    y_updated_B = np.concatenate([y, new_y_B])
    print(f"Found {len(new_X_B)} new sunglasses+dog embeddings. Updated Model_B training set size: {X_updated_B.shape[0]}")
else:
    X_updated_B = X.copy()
    y_updated_B = y.copy()
    print(f"WARNING: no images found in 'updated_samples_B/'. Add "
          f"'updated_samples_B/{TARGET_IDENTITY}/*.jpg' (sunglasses+dog photos) before continuing.")


# ========================================================================
# Original notebook cell 42: Step 12 — Update Model_B with Sunglasses + Dog Images of Sunil Grover
# ========================================================================
# CHANGED: retrain ONLY Model_B on the updated (base + sunglasses+dog) data -- plain MLP
# retrain, NO FedProx here (same reasoning as Model_A -- see Step 11).
# Model_A (already updated) and Model_C are NOT touched.
print("Retraining Model_B (plain MLP retrain, no FedProx) on updated (base + sunglasses+dog) data ...")
models["Model_B"], retrain_loss_B = train_model(
    models["Model_B"],
    X_updated_B,   # combined old (base) + new (sunglasses+dog) samples
    y_updated_B,
    epochs=RETRAIN_EPOCHS,
)

save_path_B = os.path.join(LOCAL_MODELS_DIR, "Model_B_updated.pth")
torch.save(models["Model_B"].state_dict(), save_path_B)
print("Saved updated Model_B to:", save_path_B)
print("\nModel_A and Model_C were NOT touched in this step.")


# ========================================================================
# Original notebook cell 44: Step 12.5 — Before vs After: Model_B on the Sunglasses+Dog Image
# ========================================================================
# ADDED: upload a SUNGLASSES+DOG test image of Sunil Grover (used for the before-vs-after check)
print("Please upload a SUNGLASSES+DOG test image of Sunil Grover...")
uploaded_sunglasses_check = files.upload()
sunglasses_check_image_path = list(uploaded_sunglasses_check.keys())[0]
print("Using sunglasses+dog test image:", sunglasses_check_image_path)


# ========================================================================
# Original notebook cell 45: Step 12.5 — Before vs After: Model_B on the Sunglasses+Dog Image
# ========================================================================
# ADDED: BEFORE vs AFTER comparison -- Model_B on the sunglasses+dog image
# BEFORE = baseline Model_B (saved in Step 8, prior to the Step 12 sunglasses+dog retraining)
# AFTER  = Model_B after Step 12's FedProx retraining on (base + sunglasses+dog) data

baseline_model_B = FaceMLP(num_classes=NUM_CLASSES)
baseline_model_B.load_state_dict(
    torch.load(os.path.join(LOCAL_MODELS_DIR, "Model_B_baseline.pth"), map_location=DEVICE)
)

df_before_after_B = before_after_analysis(
    "Model_B", baseline_model_B, models["Model_B"], sunglasses_check_image_path,
    "Sunglasses+Dog", label_encoder
)


# ========================================================================
# Original notebook cell 47: Step 13 — Update Model_C with Glasses Images of Sunil Grover
# ========================================================================
# ADDED: load and embed the new GLASSES images (same unchanged ArcFace pipeline)
UPDATED_DIR_C = os.path.join(BASE_DIR, "updated_samples_C")

new_X_C, new_y_C = [], []
if os.path.isdir(UPDATED_DIR_C):
    updated_people_C = [
        d for d in os.listdir(UPDATED_DIR_C)
        if os.path.isdir(os.path.join(UPDATED_DIR_C, d))
    ]

    # FIXED: match folder names to label_encoder classes CASE-INSENSITIVELY. Drive folder
    # names (e.g. "sunil grover") often don't match the exact casing label_encoder was fit
    # on (e.g. "Sunil Grover"), which used to crash deep inside sklearn with a cryptic
    # "unseen labels" KeyError. We build a lowercase -> correct-cased lookup once, and use
    # the CORRECTLY CASED label for training while still reading images from the folder's
    # actual (whatever-cased) path on disk.
    class_lookup_C = {c.lower(): c for c in label_encoder.classes_}

    unmatched_C = [p for p in updated_people_C if p.lower() not in class_lookup_C]
    if unmatched_C:
        raise ValueError(
            f"Folder name(s) {unmatched_C} under 'updated_samples_C/' don't match "
            f"any class label_encoder was trained on, even case-insensitively. Check for typos "
            f"or extra spaces. (TARGET_IDENTITY is currently '{TARGET_IDENTITY}'.)"
        )

    for person in updated_people_C:
        correct_label = class_lookup_C[person.lower()]  # ADDED: correctly-cased label
        person_dir = os.path.join(UPDATED_DIR_C, person)
        image_paths = glob.glob(os.path.join(person_dir, "*"))
        for img_path in image_paths:
            emb = extract_embedding(img_path)
            if emb is not None:
                new_X_C.append(emb)
                new_y_C.append(correct_label)  # FIXED: use correct-cased label, not raw folder name

if len(new_X_C) > 0:
    new_X_C = np.array(new_X_C, dtype=np.float32)
    new_y_C = label_encoder.transform(new_y_C)
    X_updated_C = np.vstack([X, new_X_C])
    y_updated_C = np.concatenate([y, new_y_C])
    print(f"Found {len(new_X_C)} new glasses embeddings. Updated Model_C training set size: {X_updated_C.shape[0]}")
else:
    X_updated_C = X.copy()
    y_updated_C = y.copy()
    print(f"WARNING: no images found in 'updated_samples_C/'. Add "
          f"'updated_samples_C/{TARGET_IDENTITY}/*.jpg' (glasses photos) before continuing.")


# ========================================================================
# Original notebook cell 48: Step 13 — Update Model_C with Glasses Images of Sunil Grover
# ========================================================================
# CHANGED: retrain ONLY Model_C on the updated (base + glasses) data -- plain MLP retrain,
# NO FedProx here (same reasoning as Model_A / Model_B -- see Step 11).
# Model_A (already updated) and Model_B (already updated) are NOT touched.
print("Retraining Model_C (plain MLP retrain, no FedProx) on updated (base + glasses) data ...")
models["Model_C"], retrain_loss_C = train_model(
    models["Model_C"],
    X_updated_C,   # combined old (base) + new (glasses) samples
    y_updated_C,
    epochs=RETRAIN_EPOCHS,
)

save_path_C = os.path.join(LOCAL_MODELS_DIR, "Model_C_updated.pth")
torch.save(models["Model_C"].state_dict(), save_path_C)
print("Saved updated Model_C to:", save_path_C)
print("\nModel_A and Model_B were NOT touched in this step.")


# ========================================================================
# Original notebook cell 50: Step 13.5 — Before vs After: Model_C on the Glasses Image
# ========================================================================
# ADDED: upload a GLASSES test image of Sunil Grover (used for the before-vs-after check)
print("Please upload a GLASSES test image of Sunil Grover...")
uploaded_glasses_check = files.upload()
glasses_check_image_path = list(uploaded_glasses_check.keys())[0]
print("Using glasses test image:", glasses_check_image_path)


# ========================================================================
# Original notebook cell 51: Step 13.5 — Before vs After: Model_C on the Glasses Image
# ========================================================================
# ADDED: BEFORE vs AFTER comparison -- Model_C on the glasses image
# BEFORE = baseline Model_C (saved in Step 8, prior to the Step 13 glasses retraining)
# AFTER  = Model_C after Step 13's FedProx retraining on (base + glasses) data

baseline_model_C = FaceMLP(num_classes=NUM_CLASSES)
baseline_model_C.load_state_dict(
    torch.load(os.path.join(LOCAL_MODELS_DIR, "Model_C_baseline.pth"), map_location=DEVICE)
)

df_before_after_C = before_after_analysis(
    "Model_C", baseline_model_C, models["Model_C"], glasses_check_image_path,
    "Glasses", label_encoder
)


# ========================================================================
# Original notebook cell 53: Step 14 — Share Knowledge: Full Diff-Weighted Aggregation (Global + Model_A + Model_B + Model_C)
# ========================================================================
# ADDED: Step 14 -- full 4-way diff-weighted aggregation, sharing Model_A's,
# Model_B's, and Model_C's knowledge (and the baseline Global's) into ONE new Global Model.
#
# FIXED: the diff for each participant is now measured against BASELINE_GLOBAL_REFERENCE
# (the trained baseline Global Model from Step 10), NOT ROUND0_REFERENCE (the random init).
# Reason this was a bug: Model_A/B/C all run the SAME baseline training (same data, same
# epochs) starting from ROUND0_REFERENCE before they ever see their appearance-specific
# update. That shared baseline drift is large and roughly EQUAL across all three models, so
# when the diff was measured from ROUND0_REFERENCE it was dominated by this common baseline
# drift -- the much smaller, genuinely DIFFERENT drift each model picked up from its own
# appearance update (mask / sunglasses+dog / glasses) got swamped, and all three ended up
# with nearly identical coefficients no matter how much each one actually changed.
# Measuring from BASELINE_GLOBAL_REFERENCE instead isolates exactly that appearance-update
# drift (each model's distance from the point right before its own retraining), which is
# what this aggregation step is actually supposed to weight by. As a side effect, the
# baseline Global's own diff from this same reference is ~0 (it hasn't been retrained since
# Step 10), so it correctly contributes almost nothing here -- it has nothing new to share.
all_state_dicts = [
    copy.deepcopy(global_model.state_dict()),        # Global (baseline consensus)
    copy.deepcopy(models["Model_A"].state_dict()),    # mask-updated
    copy.deepcopy(models["Model_B"].state_dict()),    # sunglasses+dog-updated
    copy.deepcopy(models["Model_C"].state_dict()),    # glasses-updated
]

# FIXED: BASELINE_GLOBAL_REFERENCE may be on CPU or GPU depending on where it was created;
# weighted_federated_average / compute_weight_diff already move tensors to a common device
# internally, so no extra device handling is needed here.
global_state_dict_final, final_coefficients = weighted_federated_average(
    all_state_dicts,
    names=["Global", "Model_A", "Model_B", "Model_C"],
    reference_state_dict=BASELINE_GLOBAL_REFERENCE,
)

global_model.load_state_dict(global_state_dict_final)
print("Global Model updated via single 4-way diff-weighted aggregation -- knowledge shared.")

GLOBAL_FINAL_PATH = os.path.join(BASE_DIR, "federated_model", "global_model_final.pth")
torch.save(global_model.state_dict(), GLOBAL_FINAL_PATH)
print("Saved final (shared-knowledge) Global Model to:", GLOBAL_FINAL_PATH)


# ========================================================================
# Original notebook cell 55: Step 15 — Distribute the Shared Global Model Back to All Three Cities
# ========================================================================
# ADDED: distribute the final shared Global Model back to Model_A, Model_B, Model_C.
# No retraining -- pure weight copy. All three become identical after this.
for name in model_names:
    models[name].load_state_dict(copy.deepcopy(global_model.state_dict()))

print("Model_A, Model_B, and Model_C have all received the shared Global Model's weights.")
print("(No retraining performed -- weight copy only.)")


# ========================================================================
# Original notebook cell 57: Step 15.5 — Convergence: FedProx Fine-tuning of Model_A/B/C Around the Shared Global Model
# ========================================================================
# ADDED: Step 15.5 -- FedProx convergence fine-tuning. Each city fine-tunes on its own
# combined (old + new) data, anchored via the FedProx proximal term to the shared Global
# Model it just received in Step 15. This is the ONLY place in the notebook where FedProx
# is actually used for training -- the earlier individual updates (Steps 11-13) intentionally
# used plain MLP retraining so each city could learn its own appearance change freely, and
# now that everyone's knowledge has been merged and redistributed, FedProx gently pulls all
# three toward converging on that shared consensus without erasing what they just received.
CONVERGENCE_EPOCHS = 5  # fewer epochs than a full retrain -- this is fine-tuning, not training from scratch

# ADDED: snapshot of the just-distributed shared Global Model, used as the FedProx proximal
# anchor for every city's convergence fine-tuning below.
SHARED_GLOBAL_REFERENCE = copy.deepcopy(global_model.state_dict())

city_data = {
    "Model_A": (X_updated_A, y_updated_A),
    "Model_B": (X_updated_B, y_updated_B),
    "Model_C": (X_updated_C, y_updated_C),
}

convergence_loss_histories = {}
for name in model_names:
    X_city, y_city = city_data[name]
    print(f"\nConverging {name} with FedProx (mu={FEDPROX_MU}) on its own (base + updated) data ...")
    models[name], conv_loss_hist = train_model_fedprox(
        models[name],
        X_city,
        y_city,
        global_ref_state_dict=SHARED_GLOBAL_REFERENCE,
        epochs=CONVERGENCE_EPOCHS,
        mu=FEDPROX_MU,
    )
    convergence_loss_histories[name] = conv_loss_hist

    save_path = os.path.join(LOCAL_MODELS_DIR, f"{name}_converged.pth")
    torch.save(models[name].state_dict(), save_path)
    print(f"Saved converged {name} to:", save_path)

print("\nAll three cities have converged around the shared Global Model via FedProx fine-tuning.")


# ========================================================================
# Original notebook cell 59: Step 16 — Evaluate Shared Knowledge: Predict All Three Appearance Images
# ========================================================================
# ADDED: upload the three appearance-changed test images of Sunil Grover
print("Please upload the MASK test image of Sunil Grover...")
uploaded_mask = files.upload()
mask_test_image_path = list(uploaded_mask.keys())[0]
print("Using mask test image:", mask_test_image_path)


# ========================================================================
# Original notebook cell 60: Step 16 — Evaluate Shared Knowledge: Predict All Three Appearance Images
# ========================================================================
print("Please upload the SUNGLASSES+DOG test image of Sunil Grover...")
uploaded_sunglasses_dog = files.upload()
sunglasses_dog_test_image_path = list(uploaded_sunglasses_dog.keys())[0]
print("Using sunglasses+dog test image:", sunglasses_dog_test_image_path)


# ========================================================================
# Original notebook cell 61: Step 16 — Evaluate Shared Knowledge: Predict All Three Appearance Images
# ========================================================================
print("Please upload the GLASSES test image of Sunil Grover...")
uploaded_glasses = files.upload()
glasses_test_image_path = list(uploaded_glasses.keys())[0]
print("Using glasses test image:", glasses_test_image_path)


# ========================================================================
# Original notebook cell 62: Step 16 — Evaluate Shared Knowledge: Predict All Three Appearance Images
# ========================================================================
# CHANGED: predict on all three appearance images using EACH of Model_A, Model_B, Model_C
# separately -- after Step 15.5's FedProx convergence fine-tuning, they are no longer
# guaranteed to be identical, so we can no longer just evaluate one and call it "the shared
# model". This also makes the result more informative: it shows whether every city model
# independently kept knowledge of ALL THREE appearance changes.
test_images = {
    "Mask": mask_test_image_path,
    "Sunglasses+Dog": sunglasses_dog_test_image_path,
    "Glasses": glasses_test_image_path,
}

all_results = {}
print("=== Per-Model Predictions After Aggregation + FedProx Convergence ===")
for name in model_names:
    print(f"\n--- {name} ---")
    all_results[name] = {}
    for label, img_path in test_images.items():
        result = predict_topk(img_path, models[name], label_encoder, k=2)
        if result is not None:
            (top_class, top_prob), (second_class, second_prob) = result
            all_results[name][label] = (top_class, top_prob, second_class, second_prob)
            print(f"{label}: Predicted = {top_class} ({top_prob * 100:.2f}%), "
                  f"Second Best = {second_class} ({second_prob * 100:.2f}%)")

# ADDED: save all results, per model, per appearance image
result_path = os.path.join(BASE_DIR, "evaluation", "shared_model_all_appearances_prediction.txt")
with open(result_path, "w") as f:
    f.write("Per-Model Predictions (Global + Model_A + Model_B + Model_C aggregated, then FedProx-converged)\n")
    f.write("=================================================================================\n\n")
    for name in model_names:
        f.write(f"{name}:\n")
        for label, (top_class, top_prob, second_class, second_prob) in all_results[name].items():
            f.write(f"  {label} image:\n")
            f.write(f"    Predicted Class: {top_class}\n")
            f.write(f"    Confidence: {top_prob * 100:.2f}%\n")
            f.write(f"    Second Best Class: {second_class}\n")
            f.write(f"    Second Best Confidence: {second_prob * 100:.2f}%\n")
        f.write("\n")
print("\nSaved:", result_path)
