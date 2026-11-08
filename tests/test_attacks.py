import torch
import numpy as np
import pytest
from fl_security.attacks.label_flip import apply_label_flip
from fl_security.attacks.sign_flip import apply_sign_flip

def test_label_flip_default():
    labels = np.array([0, 1, 9])
    # Default mapping: i -> (i + 1) % 10 => 0->1, 1->2, 9->0
    flipped = apply_label_flip(labels, num_classes=10)
    expected = np.array([1, 2, 0])
    np.testing.assert_array_equal(flipped, expected)

def test_label_flip_custom():
    labels = np.array([0, 5, 9])
    custom_mapping = {0: 9, 5: 5, 9: 0}
    flipped = apply_label_flip(labels, num_classes=10, flip_mapping=custom_mapping)
    expected = np.array([9, 5, 0])
    np.testing.assert_array_equal(flipped, expected)

def test_sign_flip_basic():
    state_dict = {
        "weight": torch.tensor([1.0, -2.0, 3.0]),
        "count": torch.tensor([5])  # integer tensor should remain unchanged
    }
    flipped = apply_sign_flip(state_dict, strength=1.0)
    
    torch.testing.assert_close(flipped["weight"], torch.tensor([-1.0, 2.0, -3.0]))
    torch.testing.assert_close(flipped["count"], torch.tensor([5]))

def test_sign_flip_strength():
    state_dict = {
        "weight": torch.tensor([2.0, -4.0])
    }
    flipped = apply_sign_flip(state_dict, strength=2.0)
    torch.testing.assert_close(flipped["weight"], torch.tensor([-4.0, 8.0]))
