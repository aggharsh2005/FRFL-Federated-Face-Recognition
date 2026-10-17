"""Compatibility note for the old monolithic runner.

Use main.py for the modular baseline workflow:
    python main.py --dataset-dir /path/to/dataset --work-dir artifacts

For the complete original experiment—including client-specific appearance updates,
diff-weighted aggregation, FedProx fine-tuning, and notebook visualizations—open
FRFL_Federated_Face_Recognition.ipynb in Google Colab and run its cells in order.
Reusable components are split into src/frfl/model.py, training.py, embeddings.py,
aggregation.py, federated.py, inference.py, and evaluation.py.
"""
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "FRFL_Federated_Face_Recognition.ipynb"


def main():
    print("The full FRFL workflow is maintained in the companion notebook:")
    print(NOTEBOOK)
    print("For a modular baseline training run, use: python main.py --help")


if __name__ == "__main__":
    main()
