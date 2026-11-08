import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from fl_security.federated.model import CIFAR10CNN
from fl_security.federated.training import train_local
from fl_security.attacks.sign_flip import apply_sign_flip
from fl_security.attacks.label_flip import apply_label_flip

class TestAttackIntegration(unittest.TestCase):
    def test_sign_flip_attack_effect(self):
        model = CIFAR10CNN()
        state_dict = model.state_dict()
        flipped_dict = apply_sign_flip(state_dict, strength=1.0)
        for k in state_dict:
            if torch.is_floating_point(state_dict[k]):
                self.assertTrue(torch.equal(flipped_dict[k], -1.0 * state_dict[k]))

    def test_label_flip_attack_effect(self):
        labels = [0, 1, 2, 9]
        flipped_labels = apply_label_flip(labels, num_classes=10)
        self.assertEqual(list(flipped_labels), [1, 2, 3, 0])

if __name__ == '__main__':
    unittest.main()
