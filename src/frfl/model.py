"""MLP classifier and reproducibility settings used by the FRFL notebook."""
import random
import numpy as np
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_EPOCHS = 30
RETRAIN_EPOCHS = 30
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
FEDPROX_MU = 0.05
SEED = 42

class FaceMLP(nn.Module):
    """MLP classifier for 512-D ArcFace embeddings."""
    def __init__(self, num_classes: int, input_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch to improve repeatability."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
