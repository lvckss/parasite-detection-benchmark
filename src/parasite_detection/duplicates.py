"""Image fingerprint for near duplicates; same method as notebooks/01_dataset_overview.ipynb."""

import numpy as np
from PIL import Image

SIZE = 32


def fingerprint(path, size=SIZE):
    """Grayscale thumbnail normalised to zero mean and unit variance."""
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32).ravel()
    return (gray - gray.mean()) / (gray.std() + 1e-6)


def correlation(a, b):
    """Pearson correlation between every row of ``a`` and every row of ``b``."""
    return a @ b.T / a.shape[1]
