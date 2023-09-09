from typing import Union, Iterable, List
from copy import copy

import numpy as np
from numpy import ndarray


class IndexManager:
    """
    Manages and index set by generating and recycling indices
    of a set of points or cells.
    """

    def __init__(self, start=0):
        self.queue = []
        self.next = start

    def generate_np(self, n: int = 1) -> Union[int, ndarray]:
        if n == 1:
            return self.generate(1)
        else:
            return np.array(self.generate(n))

    def generate(self, n: int = 1) -> Union[int, List[int]]:
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
        for a in args:
            if isinstance(a, Iterable):
                self.queue.extend(a)
            else:
                self.queue.append(a)
