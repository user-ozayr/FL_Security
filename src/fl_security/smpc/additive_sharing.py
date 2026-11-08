import numpy as np

def generate_additive_shares(secret: np.ndarray, num_parties: int):
    shape = secret.shape
    shares = [np.random.randint(-100000, 100000, size=shape, dtype=np.int64) for _ in range(num_parties - 1)]
    last_share = secret - sum(shares)
    shares.append(last_share)
    return shares

def reconstruct_additive_shares(shares: list):
    return sum(shares)
