# FRFL — Federated Face Recognition

**Developers:** [Harsh Aggarwal](https://github.com/aggharsh2005) and [Tushar Walia](https://github.com/Tushar22-tutu)


Federated-style face recognition using InsightFace/ArcFace embeddings, a PyTorch MLP classifier, three simulated clients, diff-weighted aggregation, and FedProx fine-tuning.

## Project structure

```text
FRFL-Federated-Face-Recognition/
├── FRFL_Federated_Face_Recognition.ipynb  # Complete interactive Colab workflow
├── main.py                                # CLI entry point for modular baseline training
├── utils.py                               # Shared seed and directory utilities
├── scripts/
│   └── run_experiment.py                  # Small compatibility note; no duplicated monolithic code
├── src/frfl/
│   ├── config.py                          # Defaults and standard output directories
│   ├── model.py                           # FaceMLP and training constants
│   ├── embeddings.py                      # Face embedding extraction and cache
│   ├── training.py                        # Local training and FedProx optimizer
│   ├── aggregation.py                     # Diff-weighted state-dict aggregation
│   ├── federated.py                       # Federated aggregation interface
│   ├── clients.py                          # Client-local training and FedProx helpers
│   ├── experiment.py                       # Federated round orchestration
│   ├── inference.py                       # Top-k prediction and before/after analysis
│   └── evaluation.py                      # Evaluation-facing exports
├── requirements-colab.txt
├── pyproject.toml
└── LICENSE
```

## Which file does what?

- **`main.py`**: command-line entry point; creates embeddings and trains/saves a baseline classifier.
- **`utils.py`**: shared seed and output-directory utilities.
- **`src/frfl/model.py`**: defines the MLP classifier.
- **`src/frfl/embeddings.py`**: extracts and caches 512-dimensional ArcFace embeddings.
- **`src/frfl/training.py`**: local model training and FedProx fine-tuning.
- **`src/frfl/aggregation.py` / `federated.py`**: computes model drift and combines client state dictionaries.
- **`src/frfl/inference.py` / `evaluation.py`**: top-k predictions and before/after comparisons.
- **Notebook**: complete original end-to-end experiment, including interactive image uploads, Model_A/Model_B/Model_C appearance-specific updates, final aggregation, plots, and evaluation.

## Run the modular baseline

Install the dependencies in a suitable Python/Colab environment, then run:

```bash
python main.py --dataset-dir /path/to/105_classes_pins_dataset --work-dir artifacts --epochs 30

To also run a federated round, provide all three appearance-specific update directories:

python main.py --dataset-dir /path/to/105_classes_pins_dataset --client-a-dir /path/to/updated_samples_A --client-b-dir /path/to/updated_samples_B --client-c-dir /path/to/updated_samples_C --work-dir artifacts
```

The dataset and each client update directory should contain identity-named subfolders with images inside. Client identity names must exist in the baseline dataset. Use `--ctx-id -1` for CPU or `--ctx-id 0` for the first GPU where supported.

## Run the full experiment

Open `FRFL_Federated_Face_Recognition.ipynb` in Google Colab and run the cells in order. The notebook retains the original workflow and its Google Drive paths / interactive upload prompts. Place appearance-specific training images under the configured `updated_samples_A`, `updated_samples_B`, and `updated_samples_C` folders.

The complete interactive workflow is still notebook-based; `main.py` is a modular baseline-training entry point, not a replacement for every interactive notebook step. The notebook has not been validated end-to-end in a fresh runtime as part of this refactor.

## Experiment overview

- **Embedding model:** InsightFace `buffalo_l` / ArcFace, 512-dimensional embeddings.
- **Classifier:** MLP `512 → 256 → 128 → number_of_classes`.
- **Clients:** `Model_A`, `Model_B`, and `Model_C`, with mask, sunglasses + dog, and glasses appearance updates.
- **Aggregation:** diff-weighted averaging of PyTorch model state dictionaries.
- **Convergence:** FedProx fine-tuning around the shared global model.

The clients are simulated in one runtime; this is not a distributed network deployment. Federated learning does not itself guarantee privacy.

## Data and privacy

Do not commit face images, embeddings, model checkpoints, credentials, or other sensitive biometric data. Process only data you have permission and consent to use. Embeddings and model updates can still leak information.

## License

MIT — see [LICENSE](LICENSE).
