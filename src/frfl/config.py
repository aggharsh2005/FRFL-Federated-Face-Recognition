"""Default configuration values and project directory layout."""
from pathlib import Path

SEED = 42
EMBEDDING_DIM = 512
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_FEDPROX_MU = 0.05
DEFAULT_FACE_MODEL = "buffalo_l"


def make_project_dirs(root):
    """Create and return the standard FRFL output directories."""
    root = Path(root)
    names = (
        "dataset", "embeddings", "local_models", "federated_model",
        "evaluation", "results", "updated_samples_A", "updated_samples_B",
        "updated_samples_C", "test_images",
    )
    paths = {name: root / name for name in names}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
