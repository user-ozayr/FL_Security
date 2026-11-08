import unittest
import torch
import numpy as np
from fl_security.attacks.byzantine import ByzantineAttacker

class TestByzantineAttacker(unittest.TestCase):
    def setUp(self):
        self.config = {
            "experiment": {"seed": 42},
            "attacks": {
                "enabled": True,
                "attack_type": "sign_flip",
                "compromised_fraction": 0.4,
                "activation_rounds": [1, 2],
                "strength": 1.0
            }
        }
        self.num_clients = 5
        self.attacker = ByzantineAttacker(self.config, self.num_clients)

    def test_malicious_identification(self):
        malicious_count = sum(1 for cid in range(self.num_clients) if self.attacker.is_malicious(cid, 1))
        self.assertEqual(malicious_count, 2)

    def test_activation_rounds(self):
        for cid in range(self.num_clients):
            self.assertFalse(self.attacker.is_malicious(cid, 3))

    def test_sign_flip_perturbation(self):
        global_state = {"weight": torch.tensor([0.0, 0.0])}
        local_state = {"weight": torch.tensor([1.0, -2.0])}
        malicious_cid = next(cid for cid in range(self.num_clients) if self.attacker.is_malicious(cid, 1))
        perturbed = self.attacker.perturb_update(malicious_cid, 1, global_state, local_state)
        # Delta_local = [1.0, -2.0] - [0.0, 0.0] = [1.0, -2.0]
        # Delta_malicious = -1.0 * [1.0, -2.0] = [-1.0, 2.0]
        # Perturbed = global_state + Delta_malicious = [-1.0, 2.0]
        expected = torch.tensor([-1.0, 2.0])
        self.assertTrue(torch.equal(perturbed["weight"], expected))

    def test_benign_client_unaffected(self):
        global_state = {"weight": torch.tensor([0.0, 0.0])}
        local_state = {"weight": torch.tensor([1.0, -2.0])}
        benign_cid = next(cid for cid in range(self.num_clients) if not self.attacker.is_malicious(cid, 1))
        perturbed = self.attacker.perturb_update(benign_cid, 1, global_state, local_state)
        self.assertTrue(torch.equal(perturbed["weight"], local_state["weight"]))

if __name__ == '__main__':
    unittest.main()
