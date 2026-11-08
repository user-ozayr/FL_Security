import numpy as np

def select_clients(num_clients, fraction, strategy='random', seed=None):
    if seed is not None:
        np.random.seed(seed)
    k = max(1, int(num_clients * fraction))
    all_clients = list(range(num_clients))
    
    if strategy == 'random':
        selected = np.random.choice(all_clients, k, replace=False).tolist()
    elif strategy == 'fixed':
        selected = all_clients[:k]
    else:
        selected = np.random.choice(all_clients, k, replace=False).tolist()
    return selected
