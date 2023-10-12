import numpy as np
from numpy import ndarray

__all__ = ["trimap_Q8"]


def trimap_Q8() -> ndarray:
    return np.array(
        [
            [0, 4, 7],
            [4, 1, 5],
            [5, 2, 6],
            [6, 3, 7],
            [7, 5, 6],
            [4, 5, 7],
        ],
        dtype=int,
    )
