# FRFL — Federated Face Recognition

A notebook-based research prototype exploring face recognition across three **simulated** clients with diff-weighted model aggregation and FedProx fine-tuning.

## Current contents

- `FRFL_Federated_Face_Recognition.ipynb` — experiment entry-point / setup notebook.
- `requirements-colab.txt` — key package list for Google Colab.
- `.gitignore` — excludes sensitive biometric files and generated artifacts.
- `LICENSE` — MIT License.

**Important:** the notebook currently committed is a concise entry-point rather than the complete 64-cell experimental notebook. The complete cleaned notebook is available in the repository bundle linked below in the chat. I wasn't able to transfer the full notebook through this connection.

## Experiment overview

The full experiment, as prepared, uses InsightFace `buffalo_l` / ArcFace embeddings (512 dimensions), an MLP classifier (`512 → 256 → 128 → classes`), three simulated clients (Model_A, Model_B, Model_C), client-specific appearance updates (mask, sunglasses + dog, glasses), diff-weighted aggregation, and FedProx fine-tuning.

The workflow is simulated in one Colab runtime. It is not a distributed deployment across actual machines or organizations, and federated learning alone does not guarantee privacy.

## Running in Google Colab

1. Open the complete notebook in Google Colab.
2. Mount Google Drive when prompted.
3. Place the dataset in the locations expected by the notebook, including:
   `/content/drive/MyDrive/FRFL/dataset/105_classes_pins_dataset`
   and the `updated_samples_A`, `updated_samples_B`, `updated_samples_C` folders.
4. Check that paths and package versions match your Colab runtime before running.

## Data protection

Do not commit face images, embeddings, model checkpoints, credentials, or other sensitive data. Make sure you have the rights and consent needed to process face data. Review train/test separation and evaluation methodology before interpreting results as benchmarks.

## License

See [LICENSE](LICENSE).
