import torch

def apply_sign_flip(state_dict, strength=1.0):
    flipped = {}
    for k, v in state_dict.items():
        if torch.is_floating_point(v) or v.is_complex():
            flipped[k] = -1.0 * strength * v
        else:
            flipped[k] = v
    return flipped
