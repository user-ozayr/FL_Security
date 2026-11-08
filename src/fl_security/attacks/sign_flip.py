import torch

def apply_sign_flip(arg1, arg2=None, strength=1.0):
    """
    Applies model-update sign-flip attack.
    Supports two signatures:
    1. apply_sign_flip(state_dict, strength=1.0) -> Direct sign flip (legacy mode)
    2. apply_sign_flip(global_state_dict, local_state_dict, strength=1.0) -> Delta-based sign flip
    """
    if arg2 is None:
        # Single state dict mode (legacy/tests)
        state_dict = arg1
        # If strength is passed as keyword or positional when arg2 is None, ensure correct handling
        # Note: if called as apply_sign_flip(state_dict, strength=2.0), strength is passed as keyword.
        # But what if called as apply_sign_flip(state_dict, 2.0)? Python passes 2.0 as arg2!
        # Wait! If someone calls apply_sign_flip(state_dict, 2.0), arg2 is 2.0 (a float/tensor, not a dict).
        pass

    if arg2 is None or not isinstance(arg2, dict):
        # Single state dict mode where arg2 is either None or the strength value
        state_dict = arg1
        eff_strength = arg2 if arg2 is not None else strength
        flipped = {}
        for k, v in state_dict.items():
            if torch.is_floating_point(v) or v.is_complex():
                flipped[k] = -1.0 * eff_strength * v
            else:
                flipped[k] = v
        return flipped
    else:
        # Global + Local state dict mode
        global_state_dict = arg1
        local_state_dict = arg2
        flipped = {}
        for k in global_state_dict.keys():
            g_v = global_state_dict[k]
            l_v = local_state_dict[k]
            if torch.is_floating_point(g_v) or g_v.is_complex():
                delta_local = l_v - g_v
                delta_malicious = -1.0 * strength * delta_local
                flipped[k] = g_v + delta_malicious
            else:
                flipped[k] = l_v
        return flipped
