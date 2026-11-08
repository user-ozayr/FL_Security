import numpy as np
import torch

class FixedPointEncoder:
    def __init__(self, scale=1000000):
        self.scale = scale

    def encode(self, tensor: torch.Tensor):
        scaled = torch.round(tensor * self.scale).to(torch.int64)
        return scaled

    def decode(self, encoded_tensor: torch.Tensor):
        return encoded_tensor.float() / self.scale
