"""Federated aggregation API.

The numerical aggregation implementation lives in aggregation.py; this module
provides a small named interface for the experiment runner.
"""
from .aggregation import compute_weight_diff, weighted_federated_average


def aggregate_client_models(client_state_dicts, client_names, global_state_dict):
    """Aggregate client models and return (state_dict, aggregation_weights)."""
    return weighted_federated_average(
        state_dicts=client_state_dicts,
        names=client_names,
        reference_state_dict=global_state_dict,
    )


__all__ = ["compute_weight_diff", "weighted_federated_average", "aggregate_client_models"]
