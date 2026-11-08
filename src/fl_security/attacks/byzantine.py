"""Configurable malicious-client behaviour for controlled FL experiments."""
import math
import numpy as np
import torch
from fl_security.attacks.label_flip import apply_label_flip
from fl_security.attacks.sign_flip import apply_sign_flip

class ByzantineAttacker:
    def __init__(self, config: dict, num_clients: int):
        attacks = config.get("attacks", {})
        self.enabled = bool(attacks.get("enabled", False))
        self.attack_type = attacks.get("attack_type", "sign_flip")
        self.compromised_fraction = float(attacks.get("compromised_fraction", 0.0))
        if not 0.0 <= self.compromised_fraction <= 1.0:
            raise ValueError("compromised_fraction must be in [0, 1]")
        configured_rounds = attacks.get("activation_rounds")
        self.activation_rounds = None if configured_rounds is None else set(int(x) for x in configured_rounds)
        self.strength = float(attacks.get("strength", 1.0))
        self.seed = int(config.get("experiment", {}).get("seed", 42))
        if self.attack_type not in {"sign_flip", "label_flip", "random_noise"}:
            raise ValueError(f"unsupported attack_type: {self.attack_type}")
        count = int(math.ceil(num_clients * self.compromised_fraction))
        rng = np.random.default_rng(self.seed + 20_000)
        self.malicious_clients = (
            set(int(x) for x in rng.choice(num_clients, count, replace=False))
            if count else set()
        )

    def is_malicious(self, client_id: int, round_num: int) -> bool:
        active = self.activation_rounds is None or round_num in self.activation_rounds
        return self.enabled and active and client_id in self.malicious_clients

    def perturb_update(self, client_id: int, round_num: int, global_state_dict: dict, local_state_dict: dict) -> dict:
        if not self.is_malicious(client_id, round_num):
            return local_state_dict
        if self.attack_type == "sign_flip":
            return apply_sign_flip(global_state_dict, local_state_dict, strength=self.strength)
        if self.attack_type == "random_noise":
            result = {}
            for key, value in local_state_dict.items():
                if torch.is_floating_point(value):
                    generator = torch.Generator(device=value.device)
                    generator.manual_seed(self.seed + round_num * 10_000 + client_id)
                    noise = torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)
                    result[key] = value + noise * self.strength
                else:
                    result[key] = value
            return result
        return local_state_dict

    def perturb_labels(self, client_id: int, round_num: int, labels, num_classes: int = 10) -> np.ndarray:
        values = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
        if not self.is_malicious(client_id, round_num) or self.attack_type != "label_flip":
            return np.asarray(values)
        return apply_label_flip(values, num_classes=num_classes)
