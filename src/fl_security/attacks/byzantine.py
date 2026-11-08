import numpy as np
import torch
from fl_security.attacks.sign_flip import apply_sign_flip
from fl_security.attacks.label_flip import apply_label_flip

class ByzantineAttacker:
    def __init__(self, config, num_clients):
        attacks_config = config.get("attacks", {})
        self.enabled = attacks_config.get("enabled", False)
        self.attack_type = attacks_config.get("attack_type", "sign_flip")
        self.compromised_fraction = attacks_config.get("compromised_fraction", 0.0)
        self.activation_rounds = attacks_config.get("activation_rounds", [1, 2, 3])
        self.strength = attacks_config.get("strength", 1.0)
        self.seed = config.get("experiment", {}).get("seed", 42)
        
        num_malicious = int(np.ceil(num_clients * self.compromised_fraction))
        np.random.seed(self.seed)
        self.malicious_clients = set(np.random.choice(num_clients, size=num_malicious, replace=False).tolist())

    def is_malicious(self, client_id: int, round_num: int) -> bool:
        if not self.enabled:
            return False
        if round_num not in self.activation_rounds:
            return False
        return client_id in self.malicious_clients

    def perturb_update(self, client_id: int, round_num: int, global_state_dict: dict, local_state_dict: dict) -> dict:
        if not self.is_malicious(client_id, round_num):
            return local_state_dict

        if self.attack_type == "sign_flip":
            return apply_sign_flip(global_state_dict, local_state_dict, strength=self.strength)
        elif self.attack_type == "random_noise":
            perturbed = {}
            np.random.seed(self.seed + round_num + client_id)
            for k, v in local_state_dict.items():
                if torch.is_floating_point(v):
                    noise = torch.randn_like(v) * self.strength
                    perturbed[k] = v + noise
                else:
                    perturbed[k] = v
            return perturbed
        else:
            return local_state_dict

    def perturb_labels(self, client_id: int, round_num: int, labels, num_classes=10) -> np.ndarray:
        if not self.is_malicious(client_id, round_num):
            return labels
        if self.attack_type == "label_flip":
            return apply_label_flip(labels, num_classes=num_classes)
        return labels
