import numpy as np
import torch
import math

def convert_torch_to_float(tensor):
    if torch.is_tensor(tensor):
        return float(tensor.detach().cpu().numpy().astype(np.float32))
    elif isinstance(tensor, np.ndarray) and tensor.size == 1:
        return float(tensor.astype(np.float32))
    elif isinstance(tensor, float):
        return tensor
    elif isinstance(tensor, int):
        return float(tensor)
    else:
        raise ValueError("Cannot convert tensor to float")


def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=0, start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs
    if warmup_steps > 0:
        warmup_iters = warmup_steps

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs

    return schedule


def metric_wrapper(output, target, metric_func, classes, device, raw=True):
    batch_size, channels, height, width = output.shape
    classes = torch.Tensor(classes).view(1, -1, 1, 1).to(device)

    target_hot = (target == classes).float().to(device)
    target_max = torch.argmax(target_hot, dim=1, keepdim=True).to(device)

    if raw:
        _output = output.permute(0, 2, 3, 1).reshape(-1, channels)
        _target = target_max.permute(0, 2, 3, 1).reshape(-1)
    else:
        output_max = torch.argmax(output, dim=1, keepdim=True).to(device)
        _output = output_max.permute(0, 2, 3, 1).reshape(-1)
        _target = target_max.permute(0, 2, 3, 1).reshape(-1)

    metric = metric_func(_output, _target)

    return metric
