"""Prediction and before/after evaluation helpers."""
import numpy as np
import pandas as pd
import torch
from .model import DEVICE
from .embeddings import extract_embedding

def predict_topk(image_path, model, label_encoder, face_app, k=2, device=DEVICE):
    """Predict top-k identities from a single image."""
    embedding = extract_embedding(image_path, face_app)
    if embedding is None:
        print("Could not read image or no face detected:", image_path)
        return None
    model = model.to(device)
    model.eval()
    x_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x_tensor), dim=1).cpu().numpy()[0]
    top_indices = np.argsort(probs)[::-1][:k]
    return [(label_encoder.inverse_transform([idx])[0], float(probs[idx])) for idx in top_indices]

def before_after_analysis(model_name, baseline_model, updated_model, test_image_path,
                          appearance_label, label_encoder, face_app, target_identity,
                          display_fn=print, device=DEVICE):
    """Compare baseline and updated predictions for the same test image."""
    before = predict_topk(test_image_path, baseline_model, label_encoder, face_app, k=2, device=device)
    after = predict_topk(test_image_path, updated_model, label_encoder, face_app, k=2, device=device)
    rows = []
    for stage, result in (("BEFORE", before), ("AFTER", after)):
        if result is not None and len(result) >= 2:
            (top_c, top_p), (sec_c, sec_p) = result
            rows.append({"Stage": stage, "Predicted": top_c, "Confidence (%)": round(top_p * 100, 2),
                         "2nd Best": sec_c, "2nd Confidence (%)": round(sec_p * 100, 2)})
    frame = pd.DataFrame(rows)
    print(f"=== {model_name} on {appearance_label} Image: BEFORE vs AFTER ===")
    display_fn(frame)
    if before is not None and after is not None:
        before_c, before_p = before[0]
        after_c, after_p = after[0]
        if after_c == target_identity and before_c != target_identity:
            print(f"  -> {model_name} flipped to the correct identity ({target_identity}) after the update.")
        elif after_c == target_identity and before_c == target_identity:
            delta = (after_p - before_p) * 100
            direction = "increased" if delta >= 0 else "decreased"
            print(f"  -> {model_name} was already correct; confidence {direction} by {abs(delta):.2f} points.")
        elif before_c == target_identity:
            print(f"  -> WARNING: {model_name} was correct before but wrong after the update.")
        else:
            print(f"  -> {model_name} still does not predict {target_identity} as top-1.")
    return frame
