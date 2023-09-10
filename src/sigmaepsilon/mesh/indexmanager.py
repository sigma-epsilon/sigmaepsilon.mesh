from typing import Union, Iterable, List
from copy import copy

import numpy as np
from numpy import ndarray


class IndexManager:
    """
    Manages and index set by generating and recycling indices.

    Examples
    --------
    >>> from sigmaepsilon.mesh.indexmanager import IndexManager
    >>> im = IndexManager()
    >>> im.generate()
    0
    >>> im.generate(1)
    1
    >>> im.generate(2)
    [2, 3]
    >>> im.recycle(0)
    >>> im.generate(2)
    [0, 4]
    >>> im.generate_np(2)
    array([5, 6])
    >>> im.recycle(5, 6)
    >>> im.generate_np(2)
    array([5, 6])
    >>> im.recycle([5, 6])
    >>> im.generate_np(2)
    array([5, 6])
    """

    def __init__(self, start: int = 0):
        self.queue = []
        self.next = start

    def generate_np(self, n: int = 1) -> Union[int, ndarray]:
        """
        Generates indices as NumPy arrays.
        """
        if n == 1:
            return self.generate(1)
        else:
            return np.array(self.generate(n))

    def generate(self, n: int = 1) -> Union[int, List[int]]:
        """
        Generates one or more indices.
        """
        nQ = len(self.queue)
        if nQ > 0:
            if n == 1:
                res = self.queue.pop()
            else:
                if nQ >= n:
                    res = self.queue[:n]
                    del self.queue[:n]
                else:
                    res = copy(self.queue)
                    res.extend(range(self.next, self.next + n - nQ))
                    self.queue = []
                self.next += n - nQ
        else:
            if n == 1:
                res = self.next
            else:
                res = list(range(self.next, self.next + n))
            self.next += n
        return res

    def recycle(self, *args) -> None:
        """
        Recycles some indices.
        """
        for a in args:
            if isinstance(a, Iterable):
                self.queue.extend(a)
            else:
                self.queue.append(a)
