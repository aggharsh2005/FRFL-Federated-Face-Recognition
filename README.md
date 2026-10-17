# FRFL — Federated Face Recognition

A notebook-based prototype for face recognition with ArcFace embeddings, a compact MLP classifier, three simulated clients, diff-weighted aggregation, and FedProx fine-tuning.

## Repository structure

```text
FRFL-Federated-Face-Recognition/
├── FRFL_Federated_Face_Recognition.ipynb   # Full step-by-step notebook (64 cells)
├── scripts/
│   └── run_experiment.py                   # Ordered notebook workflow as Python code
├── src/frfl/
│   ├── __init__.py
│   ├── model.py                            # FaceMLP, settings, random seed helper
│   ├── training.py                         # Local training and FedProx training
│   ├── embeddings.py                       # ArcFace embedding extraction + cache
│   ├── aggregation.py                      # Diff-weighted aggregation helpers
│   └── inference.py                        # Top-k prediction and before/after comparison
├── requirements-colab.txt
├── pyproject.toml
├── .gitignore
└── LICENSE
```

## Model and experiment

- **Embedding model:** InsightFace `buffalo_l` / ArcFace, producing 512-dimensional face embeddings.
- **Classifier:** MLP `512 → 256 → 128 → number_of_classes`.
- **Clients:** `Model_A`, `Model_B`, and `Model_C`, each receiving a different appearance update (mask, sunglasses + dog, glasses).
- **Aggregation:** diff-weighted averaging of PyTorch state dictionaries.
- **Convergence:** FedProx fine-tuning around the shared global model.

The three clients are simulated in one Colab runtime; this repository does not implement a distributed network deployment. Federated learning by itself does not guarantee privacy.

## Run in Google Colab

1. Open `FRFL_Federated_Face_Recognition.ipynb` and run cells in order, **or** run `scripts/run_experiment.py` in a Colab runtime.
2. Install dependencies with `!pip install -r requirements-colab.txt`. If installation fails, check package compatibility with the current Colab Python/CUDA runtime.
3. Mount Google Drive when prompted. The base dataset path expected by the experiment is `MyDrive/FRFL/dataset/105_classes_pins_dataset/`.
4. Put appearance-specific training images under `MyDrive/FRFL/updated_samples_A/`, `updated_samples_B/`, and `updated_samples_C/`, using identity-named subfolders.
5. Follow the interactive prompts to upload plain, mask, sunglasses+dog, and glasses test images.

The runner preserves the original notebook's interactive Google Colab upload workflow. It is not intended as a standalone local script without adapting Drive and upload sections. The reusable modules in `src/frfl/` are separated for reuse; the notebook and runner preserve the original end-to-end sequence.

## Outputs

The experiment saves cached embeddings, model checkpoints, and evaluation results under `MyDrive/FRFL/` in folders such as `embeddings/`, `local_models/`, `federated_model/`, and `evaluation/`. These generated artifacts are intentionally excluded from Git.

## Data and privacy

- Do not commit face images, embeddings, checkpoints, credentials, or other sensitive biometric data.
- Use only data you have permission and consent to process.
- Model updates and embeddings can still leak information; do not treat federated learning as a privacy guarantee.
- Review train/test separation and validate results before presenting them as benchmarks.

## Reproducibility

Record runtime/package versions, confirm dataset labels and paths, run from a clean runtime, and evaluate on a held-out test set. The notebook has not been executed end-to-end as part of this restructuring.

## License

MIT — see [LICENSE](LICENSE).
