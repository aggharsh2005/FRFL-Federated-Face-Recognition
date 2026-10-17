"""Composable federated-round orchestration for FRFL."""
import copy
from .model import DEVICE
from .clients import train_client, fine_tune_client
from .federated import aggregate_client_models


def run_federated_round(global_model, client_datasets, epochs=30, learning_rate=1e-3):
    """Train each named client, aggregate client updates, and return round artifacts.

    Args:
        global_model: initialized FaceMLP model.
        client_datasets: mapping like {"Model_A": (X_a, y_a), "Model_B": (X_b, y_b)}.
        epochs: local epochs for each client.
        learning_rate: local Adam learning rate.

    Returns:
        (updated_global_model, client_models, aggregation_weights, client_histories)
    """
    if not client_datasets:
        raise ValueError("client_datasets must contain at least one client")
    reference_state = copy.deepcopy(global_model.state_dict())
    client_models, histories, state_dicts, names = {}, {}, [], []
    for name, (X_client, y_client) in client_datasets.items():
        model, history = train_client(global_model, X_client, y_client,
                                      epochs=epochs, lr=learning_rate)
        client_models[name] = model
        histories[name] = history
        names.append(name)
        state_dicts.append(copy.deepcopy(model.state_dict()))

    aggregated_state, weights = aggregate_client_models(
        client_state_dicts=state_dicts,
        client_names=names,
        global_state_dict=reference_state,
    )
    updated_global = copy.deepcopy(global_model).to(DEVICE)
    updated_global.load_state_dict(aggregated_state)
    return updated_global, client_models, dict(zip(names, weights)), histories


def fine_tune_clients_around_global(client_models, client_datasets, global_model,
                                    epochs=30, learning_rate=1e-3, mu=0.05):
    """FedProx fine-tune each client against the current shared global model."""
    reference_state = copy.deepcopy(global_model.state_dict())
    tuned_models, histories = {}, {}
    for name, model in client_models.items():
        if name not in client_datasets:
            raise KeyError(f"Missing training data for client: {name}")
        X_client, y_client = client_datasets[name]
        tuned_models[name], histories[name] = fine_tune_client(
            model, X_client, y_client, global_reference=reference_state,
            epochs=epochs, lr=learning_rate, mu=mu,
        )
    return tuned_models, histories
