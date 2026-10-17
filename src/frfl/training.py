"""Local training and FedProx optimization helpers."""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from .model import DEVICE, BATCH_SIZE, LEARNING_RATE

def make_dataloader(X_data, y_data, batch_size=BATCH_SIZE, shuffle=True):
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.long)
    return DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)

def train_model(model, X_data, y_data, epochs, lr=LEARNING_RATE):
    """Train a model with cross-entropy and Adam."""
    model = model.to(DEVICE)
    model.train()
    dataloader = make_dataloader(X_data, y_data)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")
    return model, loss_history

def train_model_fedprox(model, X_data, y_data, global_ref_state_dict, epochs, lr=LEARNING_RATE, mu=0.01):
    """Train locally with a FedProx proximal penalty against reference weights."""
    model = model.to(DEVICE)
    model.train()
    global_ref_params = {name: param.detach().clone().to(DEVICE) for name, param in global_ref_state_dict.items()}
    dataloader = make_dataloader(X_data, y_data)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            ce_loss = criterion(model(batch_X), batch_y)
            proximal_term = 0.0
            for name, param in model.named_parameters():
                proximal_term += torch.sum((param - global_ref_params[name]) ** 2)
            loss = ce_loss + (mu / 2) * proximal_term
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs} - loss (CE + proximal): {avg_loss:.4f}")
    return model, loss_history
