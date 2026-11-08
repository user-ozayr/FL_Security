import numpy as np

def apply_label_flip(labels, num_classes=10, flip_mapping=None):
    if flip_mapping is None:
        flip_mapping = {i: (i + 1) % num_classes for i in range(num_classes)}
    return np.array([flip_mapping.get(int(l), l) for l in labels])
