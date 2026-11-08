import unittest
import torch
import numpy as np
from fl_security.smpc.fixed_point import FixedPointEncoder
from fl_security.smpc.additive_sharing import generate_additive_shares, reconstruct_additive_shares

class TestSMPC(unittest.TestCase):
    def test_fixed_point_encoding(self):
        """Verify fixed-point encoder and decoder precision."""
        encoder = FixedPointEncoder(scale=1000)
        tensor = torch.tensor([1.234, -5.678])
        encoded = encoder.encode(tensor)
        decoded = encoder.decode(encoded)
        self.assertTrue(torch.allclose(tensor, decoded, atol=1e-3))

    def test_additive_sharing(self):
        """Verify additive secret sharing generation and reconstruction."""
        secret = np.array([10, 20, 30], dtype=np.int64)
        num_parties = 3
        shares = generate_additive_shares(secret, num_parties)
        
        self.assertEqual(len(shares), num_parties)
        reconstructed = reconstruct_additive_shares(shares)
        np.testing.assert_array_equal(secret, reconstructed)

if __name__ == '__main__':
    unittest.main()
