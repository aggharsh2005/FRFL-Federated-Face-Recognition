"""Client-local training helpers for the three simulated FRFL participants."""
import copy
from .model import FaceMLP, DEVICE, TRAIN_EPOCHS, RETRAIN_EPOCHS, LEARNING_RATE, FEDPROX_MU
from .training import train_model, train_model_fedprox


def train_client(global_model, X_client, y_client, epochs=TRAIN_EPOCHS, lr=LEARNING_RATE):
    """Start a client from the global weights and train on that client's data."""
    local_model = copy.deepcopy(global_model).to(DEVICE)
    local_model.load_state_dict(global_model.state_dict())
    local_model, loss_history = train_model(local_model, X_client, y_client, epochs=epochs, lr=lr)
    return local_model, loss_history


def fine_tune_client(model, X_client, y_client, global_reference, epochs=RETRAIN_EPOCHS,
                     lr=LEARNING_RATE, mu=FEDPROX_MU):
    """Fine-tune a client with FedProx regularization around shared global weights."""
    local_model = copy.deepcopy(model).to(DEVICE)
    local_model, loss_history = train_model_fedprox(
        local_model, X_client, y_client,
        global_ref_state_dict=global_reference,
        epochs=epochs, lr=lr, mu=mu,
    )
    return local_model, loss_history
