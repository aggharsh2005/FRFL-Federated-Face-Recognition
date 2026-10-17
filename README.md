# Federated Face Recognition with Federated Learning (FRFL)

A notebook-based prototype exploring federated face recognition across three simulated clients. The workflow uses InsightFace embeddings, a lightweight neural-network classifier, client-specific model updates, weighted aggregation, and FedProx-style fine-tuning.

> **Status:** Research / learning prototype. The notebook has not been validated as a production-ready privacy-preserving face-recognition system. Run and review it carefully before drawing performance or privacy conclusions.

## Project overview

The notebook demonstrates a pipeline built around:
- **Face embeddings:** InsightFace `buffalo_l` / ArcFace features (512-dimensional embeddings).
- **Classifier:** An MLP with layers `512 → 256 → 128 → number_of_classes`.
- **Simulated clients:** Model_A, Model_B, and Model_C with separate client-specific sample updates (for example, appearance variations such as masks or glasses).
- **Aggregation:** Difference/weight-based aggregation to update a global model.
- **Fine-tuning:** A FedProx-style step to explore local optimization under federated training.
- **Evaluation:** Compare the baseline global model with the updated model using the notebook's evaluation cells.

This is a simulated federated workflow, not evidence that a multi-device or multi-organization deployment has been implemented.

## Repository structure

```text
FRFL-Federated-Face-Recognition/
├── FRFL_Federated_Face_Recognition.ipynb
├── README.md
├── requirements-colab.txt
├── .gitignore
└── LICENSE
```

## Run in Google Colab

1. Open `FRFL_Federated_Face_Recognition.ipynb` in Google Colab.
2. Set the runtime according to your available hardware. A GPU runtime may help, but availability and package compatibility can vary.
3. Mount Google Drive when prompted.
4. Place the dataset in the directory expected by the notebook (see below).
5. Run the notebook cells in order. Check the paths and review each stage's output before proceeding.

## Dataset and paths

The notebook expects data to be available under Google Drive paths similar to:

```text
MyDrive/FRFL/dataset/105_classes_pins_dataset/
MyDrive/FRFL/updated_samples_A/
MyDrive/FRFL/updated_samples_B/
MyDrive/FRFL/updated_samples_C/
```

Use the exact folder and file layout expected by the notebook cells; adjust the notebook paths if your dataset is organized differently.

**Dataset note:** This repository does not include the face dataset or any personal face images. Ensure that you have the rights and consent required to use any face data. Do not commit identifiable face images, embeddings, credentials, or other sensitive data.

## Dependencies

The `requirements-colab.txt` file lists the key packages used for the notebook. Colab images and GPU/ONNX Runtime compatibility change over time; if installation fails, check the current package support and the runtime's Python/CUDA setup rather than assuming the pinned versions work everywhere.

## Important caveats

- Federated learning by itself does **not** guarantee privacy. Model updates and embeddings can still leak information; consider appropriate threat models and privacy protections before deployment.
- The three clients are simulated within a notebook. This is not a distributed client/server deployment.
- Results depend on the dataset, identity split, preprocessing, random seeds, class balance, and evaluation protocol.
- Review the notebook for data leakage and ensure the train/test split reflects the intended real-world setting.
- Do not interpret notebook outputs as independently verified benchmarks.

## Reproducibility checklist

- [ ] Confirm dataset permission, structure, and label mapping.
- [ ] Check all Google Drive paths before running.
- [ ] Record the Colab runtime and package versions.
- [ ] Run the notebook from top to bottom with a clean runtime.
- [ ] Report evaluation metrics on a held-out test set.
- [ ] Keep datasets, embeddings, checkpoints, and generated outputs out of Git.

## License

This project is released under the MIT License; see [LICENSE](LICENSE).
