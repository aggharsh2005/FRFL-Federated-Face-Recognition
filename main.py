"""CLI entry point for modular FRFL baseline and federated training."""
from pathlib import Path
import argparse
import torch
from sklearn.preprocessing import LabelEncoder
from insightface.app import FaceAnalysis

from utils import set_global_seed, ensure_directories
from src.frfl.model import FaceMLP, DEVICE, TRAIN_EPOCHS, RETRAIN_EPOCHS, LEARNING_RATE, FEDPROX_MU
from src.frfl.embeddings import build_or_load_embedding_dataset
from src.frfl.training import train_model
from src.frfl.experiment import run_federated_round, fine_tune_clients_around_global


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create ArcFace embeddings, train a baseline, and optionally run a federated round."
    )
    parser.add_argument("--dataset-dir", required=True,
                        help="Training dataset with one subfolder per identity.")
    parser.add_argument("--work-dir", default="artifacts",
                        help="Output directory for cached embeddings and checkpoints.")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS,
                        help="Local training epochs.")
    parser.add_argument("--fedprox-epochs", type=int, default=RETRAIN_EPOCHS,
                        help="FedProx fine-tuning epochs.")
    parser.add_argument("--ctx-id", type=int, default=-1,
                        help="InsightFace device id: -1 for CPU, 0 for first GPU.")
    parser.add_argument("--client-a-dir", help="Optional Model_A update-image directory.")
    parser.add_argument("--client-b-dir", help="Optional Model_B update-image directory.")
    parser.add_argument("--client-c-dir", help="Optional Model_C update-image directory.")
    return parser.parse_args()


def discover_classes(directory):
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    classes = sorted(p.name for p in path.iterdir() if p.is_dir())
    if not classes:
        raise ValueError(f"No identity subfolders found in {path}")
    return classes


def main():
    args = parse_args()
    set_global_seed()
    work_dir = Path(args.work_dir)
    embedding_root = work_dir / "embeddings"
    model_root = work_dir / "models"
    ensure_directories(embedding_root, model_root)

    dataset_dir = Path(args.dataset_dir)
    class_names = discover_classes(dataset_dir)
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)

    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=args.ctx_id)

    X, y, label_names = build_or_load_embedding_dataset(
        dataset_dir=str(dataset_dir),
        embedding_dir=str(embedding_root / "baseline"),
        class_names=class_names,
        label_encoder=label_encoder,
        face_app=face_app,
    )
    if len(X) == 0:
        raise ValueError("No face embeddings were generated from the baseline dataset.")

    global_model = FaceMLP(num_classes=len(label_names)).to(DEVICE)
    global_model, baseline_losses = train_model(global_model, X, y, epochs=args.epochs)
    torch.save(global_model.state_dict(), model_root / "baseline_model.pt")
    print(f"Saved baseline model: {model_root / 'baseline_model.pt'}")

    client_dirs = {
        "Model_A": args.client_a_dir,
        "Model_B": args.client_b_dir,
        "Model_C": args.client_c_dir,
    }
    provided = [value for value in client_dirs.values() if value]
    if provided and len(provided) != 3:
        raise ValueError("Provide all three client directories, or none of them.")

    if len(provided) == 3:
        client_datasets = {}
        for client_name, directory in client_dirs.items():
            update_classes = discover_classes(directory)
            unknown = sorted(set(update_classes) - set(label_encoder.classes_))
            if unknown:
                raise ValueError(
                    f"{client_name} contains identities absent from the baseline dataset: {unknown}"
                )
            X_client, y_client, _ = build_or_load_embedding_dataset(
                dataset_dir=directory,
                embedding_dir=str(embedding_root / client_name.lower()),
                class_names=update_classes,
                label_encoder=label_encoder,
                face_app=face_app,
            )
            if len(X_client) == 0:
                raise ValueError(f"No face embeddings were generated for {client_name}.")
            client_datasets[client_name] = (X_client, y_client)

        global_model, client_models, weights, _ = run_federated_round(
            global_model, client_datasets, epochs=args.epochs, learning_rate=LEARNING_RATE
        )
        torch.save(global_model.state_dict(), model_root / "global_federated_model.pt")
        print("Aggregation weights:", weights)

        tuned_models, _ = fine_tune_clients_around_global(
            client_models, client_datasets, global_model,
            epochs=args.fedprox_epochs, learning_rate=LEARNING_RATE, mu=FEDPROX_MU,
        )
        for client_name, model in tuned_models.items():
            path = model_root / f"{client_name.lower()}_fedprox.pt"
            torch.save(model.state_dict(), path)
            print(f"Saved fine-tuned client model: {path}")

    print("Training complete. Device:", DEVICE)


if __name__ == "__main__":
    main()
