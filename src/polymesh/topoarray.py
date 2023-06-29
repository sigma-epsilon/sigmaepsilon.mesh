from typing import Iterable
import numpy as np

from awkward import Array as akArray

from neumann.linalg.sparse import JaggedArray
from neumann.arraysetops import unique2d
from neumann import atleast2d


__all__ = ["TopologyArray"]


HANDLED_FUNCTIONS = {}


class TopologyArray(JaggedArray):
    """
    A class to handle complex topologies. It is a subclass of
    the JaggedArray class from the Neumann library and is compatible
    with Numpy universal functions.

    Parameters
    ----------
    *topo: Iterable
        One or more 2d arrays definig topologies for polygonal cells.
    cuts: Iterable, Optional
        An iterable that tells how to unflatten an 1d array into a
        2d jagged shape. Only if topology is provided as a 1d array.
        Default is None.
    force_numpy: bool, Optional
        Forces dense inputs to be NumPy arrays in the background.
        Default is True.

    Examples
    --------
    The following could be the definiton for a mesh consisting from
    two line cells, one with 3 nodes and another with 2:

    >>> import numpy as np
    >>> from polymesh import TopologyArray
    >>> data = np.array([0, 1, 2, 3, 4])
    >>> TopologyArray(data, cuts=[3, 2])
    TopologyArray([[0, 1, 2], [3, 4]])

    The same mesh defined in another way:

    >>> topo1 = np.array([0, 1, 2])
    >>> topo2 = np.array([3, 4])
    >>> TopologyArray(topo1, topo2)
    TopologyArray([[0, 1, 2], [3, 4]])

    Let assume we have two 4-noded quadrilaterals as well:

    >>> topo3 = np.array([[5, 6, 7, 8],[6, 7, 9, 10]])
    >>> TopologyArray(topo1, topo2, topo3)
    TopologyArray([[0, 1, 2], [3, 4], [5, 6, 7, 8], [6, 7, 9, 10]])

    Since the TopologyArray class is a subclass of JaggedArray, we
    can easily transform it to a CSR matrix, or an Awkward array:

    >>> TopologyArray(topo1, topo2, topo3).to_csr()
    >>> TopologyArray(topo1, topo2, topo3).to_ak()
    <Array [[0, 1, 2], [3, 4, ... [6, 7, 9, 10]] type='4 * var * int32'>

    To get the unique indices in a mesh, you can simply use NumPy:

    >>> t = TopologyArray(topo1, topo2, topo3)
    >>> np.unique(t)
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

    You can also combine different topologies into one object by
    stacking:

    >>> t1 = TopologyArray(topo1, topo2)
    >>> t2 = TopologyArray(topo3)
    >>> np.vstack([t1, t2])
    TopologyArray([[0, 1, 2], [3, 4], [5, 6, 7, 8], [6, 7, 9, 10]])

    See Also
    --------
    :class:`~neumann.linalg.sparse.JaggedArray`
    """

    def __init__(self, *topo, cuts: Iterable = None, force_numpy: bool = True):
        if len(topo) == 1 and cuts is None:
            if isinstance(topo[0], np.ndarray):
                data = atleast2d(topo[0], front=True)
            elif isinstance(topo[0], akArray):
                data = topo[0]
        elif len(topo) == 1 and cuts is not None:
            data = np.array(topo[0]).astype(int)
            cuts = np.array(cuts).astype(int)
        else:
            topo = list(map(lambda t: atleast2d(t, front=True), topo))
            widths = list(map(lambda topo: topo.shape[1], topo))
            widths = np.array(widths, dtype=int)
            cN, cE = 0, 0
            for i in range(len(topo)):
                dE = topo[i].shape[0]
                cE += dE
                cN += dE * topo[i].shape[1]
            data = np.zeros(cN, dtype=int)
            cuts = np.zeros(cE, dtype=int)
            cN, cE = 0, 0
            for i in range(len(topo)):
                dE = topo[i].shape[0]
                dN = dE * topo[i].shape[1]
                data[cN : cN + dN] = topo[i].flatten()
                cN += dN
                cuts[cE : cE + dE] = np.full(dE, widths[i])
                cE += dE
        super(TopologyArray, self).__init__(data, cuts=cuts, force_numpy=force_numpy)

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            arrs = [arg._wrapped for arg in args]
            return func(*arrs, **kwargs)
        # Note: this allows subclasses that don't override
        # __array_function__ to handle DiagonalArray objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


def implements(numpy_function):
    """
    Register an __array_function__ implementation for
    TopologyArray objects.
    """

    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


@implements(np.unique)
def unique(*args, **kwargs):
    return unique2d(args[0]._wrapped, **kwargs)


@implements(np.vstack)
def vstack(*args, **kwargs):
    data = np.concatenate(list(t.flatten() for t in args[0]))
    cuts = np.concatenate(list(t.widths() for t in args[0]))
    return TopologyArray(data, cuts=cuts)
