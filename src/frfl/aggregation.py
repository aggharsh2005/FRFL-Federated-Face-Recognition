"""Diff-weighted aggregation of PyTorch model state dictionaries."""
import copy
import torch

def compute_weight_diff(state_dict, reference_state_dict):
    """Return the L2 distance between state-dict tensors and a reference."""
    total_diff = 0.0
    for key in state_dict.keys():
        current_tensor = state_dict[key].float()
        reference_tensor = reference_state_dict[key].float().to(current_tensor.device)
        total_diff += torch.sum((current_tensor - reference_tensor) ** 2).item()
    return total_diff ** 0.5

def weighted_federated_average(state_dicts, names, reference_state_dict, epsilon=1e-6):
    """Average state dictionaries with weights proportional to reference drift."""
    if not state_dicts:
        raise ValueError("state_dicts must contain at least one participant")
    if len(state_dicts) != len(names):
        raise ValueError("names and state_dicts must have the same length")
    diffs = [compute_weight_diff(sd, reference_state_dict) for sd in state_dicts]
    adjusted = [d + epsilon for d in diffs]
    total = sum(adjusted)
    coefficients = [a / total for a in adjusted]
    print("Diff-weighted aggregation coefficients:")
    for name, diff, coef in zip(names, diffs, coefficients):
        print(f"  {name}: weight difference = {diff:.6f}  ->  aggregation coefficient = {coef:.4f}")
    target_device = torch.device("cpu")
    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state.keys():
        avg_state[key] = coefficients[0] * state_dicts[0][key].float().to(target_device)
        for i in range(1, len(state_dicts)):
            avg_state[key] = avg_state[key] + coefficients[i] * state_dicts[i][key].float().to(target_device)
    return avg_state, coefficients
