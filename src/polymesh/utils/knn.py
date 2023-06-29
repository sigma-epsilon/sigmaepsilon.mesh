import numpy as np
from numpy import ndarray
from numba import njit, prange
import scipy as sp
from packaging import version
import warnings

try:
    from sklearn.neighbors import KDTree

    __has_sklearn__ = True
except Exception:
    __has_sklearn__ = False

__scipy_version__ = sp.__version__
__cache = True


def k_nearest_neighbours(
    X: ndarray,
    Y: ndarray = None,
    *,
    backend: str = "scipy",
    k: int = 1,
    workers: int = -1,
    tree_kwargs: dict = None,
    query_kwargs: dict = None,
    leaf_size: int = 30,
    return_distance: bool = False,
    max_distance: float = None,
):
    """
    Returns the k nearest neighbours (KNN) of a KDTree for a pointcloud using `scipy`
    or `sklearn`. The function acts as a uniform interface for similar functionality
    of `scipy` and `sklearn`. The most important parameters are highlighted, for the
    complete list of arguments, see the corresponding docs:

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree

    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html

    To learn more about nearest neighbour searches in general:

    https://scikit-learn.org/stable/modules/neighbors.html

    Parameters
    ----------
    X: numpy.ndarray
        An array of points to build the tree.
    Y: numpy.ndarray, Optional
        An array of sampling points to query the tree. If None it is the
        same as the points used to build the tree. Default is None.
    k: int or Sequence[int], Optional
        Either the number of nearest neighbors to return,
        or a list of the k-th nearest neighbors to return, starting from 1.
    leaf_size: positive int, Optional
        The number of points at which the algorithm switches over to brute-force.
        Default is 10.
    workers: int, Optional
        Only if backend is 'scipy'.
        Number of workers to use for parallel processing. If -1 is given all
        CPU threads are used. Default: -1.
        New in 'scipy' version 1.6.0.
    max_distance: float, Optional
        Return only neighbors within this distance. It can be a single value, or
        an array of values of shape matching the input, while a None value
        translates to an infinite upper bound.
        Default is None.
    tree_kwargs: dict, Optional
        Extra keyword arguments passed to the KDTree creator of the selected
        backend. Default is None.

    Returns
    -------
    d: float or array of floats
        The distances to the nearest neighbors. Only returned if
        `return_distance==True`.
    i: integer or array of integers
        The index of each neighbor.

    Raises
    ------
    ImportError
        In the abscence of a usable backend.

    Examples
    --------
    >>> from polymesh.grid import Grid
    >>> from polymesh import KNN
    >>> size = 80, 60, 20
    >>> shape = 10, 8, 4
    >>> grid = Grid(size=size, shape=shape, eshape='H8')
    >>> X = grid.centers()
    >>> i = KNN(X, X, k=3, max_distance=10.0)
    """
    tree_kwargs = {} if tree_kwargs is None else tree_kwargs
    query_kwargs = {} if query_kwargs is None else query_kwargs
    if backend == "scipy":
        from scipy.spatial import KDTree

        tree = KDTree(X, leafsize=leaf_size, **tree_kwargs)
        max_distance = np.inf if max_distance is None else max_distance
        query_kwargs["distance_upper_bound"] = max_distance
        if version.parse(__scipy_version__) < version.parse("1.6.0"):
            warnings.warn(
                "Multithreaded execution of a KNN search is "
                + "running on a single thread in scipy<1.6.0. Install a newer"
                + "version or use `backend=sklearn` if scikit is installed."
            )
            d, i = tree.query(Y, k=k, **query_kwargs)
        else:
            d, i = tree.query(Y, k=k, workers=workers)
    elif backend == "sklearn":
        if not __has_sklearn__:
            raise ImportError("'sklearn' must be installed for this!")

        tree = KDTree(X, leaf_size=leaf_size, **tree_kwargs)
        if max_distance is None:
            d, i = tree.query(Y, k=k, **query_kwargs)
        else:
            r = max_distance
            d, i = tree.query_radius(Y, r, k=k, **query_kwargs)
    else:
        raise ImportError("Either `sklearn` or `scipy` must be present for this!")
    return (d, i) if return_distance else i


@njit(nogil=True, parallel=True, cache=__cache)
def knn_to_lines(inds: ndarray):
    nN, nK = inds.shape
    res = np.zeros((nN, nK, 2), dtype=inds.dtype)
    for i in prange(nN):
        for j in prange(nK):
            res[i, j, 0] = i
            res[i, j, 1] = inds[i, j]
    return res
