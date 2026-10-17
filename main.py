"""Entry point for the modular FRFL Python workflow.

The original end-to-end experiment (including the three appearance-specific client
updates, notebook prompts, plots, and final comparisons) remains in the companion
Colab notebook. This entry point builds embeddings and trains/saves a baseline
classifier using the reusable modules under src/frfl.
"""
from pathlib import Path
import argparse
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from insightface.app import FaceAnalysis

from src.frfl.model import FaceMLP, DEVICE, TRAIN_EPOCHS\nfrom utils import set_global_seed, ensure_directories
from src.frfl.embeddings import build_or_load_embedding_dataset
from src.frfl.training import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Build FRFL embeddings and train a baseline model.")
    parser.add_argument("--dataset-dir", required=True,
                        help="Directory containing one subfolder per identity.")
    parser.add_argument("--work-dir", default="artifacts",
                        help="Directory for cached embeddings and model weights.")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--ctx-id", type=int, default=-1,
                        help="InsightFace device id: -1 for CPU, 0 for first GPU.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed()
    work_dir = Path(args.work_dir)
    embeddings_dir = work_dir / "embeddings"
    models_dir = work_dir / "models"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    class_names = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    if len(class_names) < 2:
        raise ValueError("Dataset must contain at least two identity subfolders.")

    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=args.ctx_id)

    X, y, label_names = build_or_load_embedding_dataset(
        dataset_dir=str(dataset_dir),
        embedding_dir=str(embeddings_dir),
        class_names=class_names,
        label_encoder=label_encoder,
        face_app=face_app,
    )
    if len(X) == 0:
        raise ValueError("No face embeddings were generated. Check the dataset images.")

    model = FaceMLP(num_classes=len(label_names)).to(DEVICE)
    model, loss_history = train_model(model, X, y, epochs=args.epochs)
    model_path = models_dir / "baseline_model.pt"
    np.save(models_dir / "label_names.npy", label_names)
    import torch
    torch.save(model.state_dict(), model_path)
    print(f"Baseline model saved to: {model_path}")
    print(f"Final training loss: {loss_history[-1]:.4f}" if loss_history else "No training epochs ran.")


if __name__ == "__main__":
    main()
