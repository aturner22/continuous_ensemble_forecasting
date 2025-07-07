import torch

def materialise_batches(loader, device, num_variables, max_horizon, latitude, longitude):
    batches = []
    for previous_fields, current_fields, valid_time in loader:
        previous_fields = previous_fields.to(device)
        current_fields = current_fields.view(-1, num_variables, len(latitude), len(longitude)).to(device)
        time_normalised = torch.tensor([valid_time[0]], dtype=torch.float32, device=device) / max_horizon
        batches.append((previous_fields, current_fields, time_normalised))
    return batches
