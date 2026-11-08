"""Fixed-point conversion for model parameters shared over a finite field."""
from __future__ import annotations
import numpy as np
import torch
from fl_security.smpc.additive_sharing import DEFAULT_FIELD_MODULUS, validate_modulus

class FixedPointEncoder:
    def __init__(self, scale: int = 1_000_000):
        if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
            raise ValueError("scale must be a positive integer")
        self.scale = scale

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            raise TypeError("encode expects a floating-point torch.Tensor")
        value = tensor.detach().to(dtype=torch.float64, device="cpu")
        if not torch.isfinite(value).all():
            raise ValueError("tensor contains NaN or infinite values")
        scaled = torch.round(value * self.scale)
        if torch.any(torch.abs(scaled) >= float(np.iinfo(np.int64).max)):
            raise OverflowError("fixed-point value exceeds int64 range")
        return scaled.to(dtype=torch.int64)

    def decode(self, encoded_tensor: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(encoded_tensor) or encoded_tensor.dtype != torch.int64:
            raise TypeError("decode expects an int64 torch.Tensor")
        return encoded_tensor.to(dtype=torch.float32) / self.scale

    def decode_modular(self, residues, modulus: int = DEFAULT_FIELD_MODULUS) -> torch.Tensor:
        """Decode Z_q residues with centred signed-integer representation."""
        q = validate_modulus(modulus)
        array = residues.detach().cpu().numpy() if torch.is_tensor(residues) else np.asarray(residues)
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError("residues must contain integers")
        centred = np.remainder(array, q)
        signed = np.where(centred > q // 2, centred - q, centred).astype(np.int64)
        return torch.from_numpy(signed.copy()).to(torch.float64) / self.scale
