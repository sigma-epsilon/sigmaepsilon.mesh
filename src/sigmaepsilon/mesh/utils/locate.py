from typing import Tuple

import numpy as np
from numpy import ndarray

from neumann import atleast2d

from .knn import k_nearest_neighbours
from .utils import cell_centers_bulk
from .cells.utils import (
    _find_first_hits_,
    _find_first_hits_knn_,
    _ntri_to_loc_bulk_,
)
from .tri import _glob_to_nat_tri_bulk_knn_, __pip_tri_bulk__, _glob_to_nat_tri_bulk_
from ..utils.utils import points_of_cells


def locate_tri_2d(
    x: ndarray,
    coords: ndarray,
    triangles: ndarray,
    trimap: ndarray = None,
    lazy: bool = True,
    k: int = 4,
    tol: float = 1e-12,
) -> Tuple[ndarray]:
    x = atleast2d(x, front=True)

    if trimap is None:
        trimap = np.array([[0, 1, 2]], dtype=int)
    lcoords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    ecoords_tri = points_of_cells(coords, triangles, centralize=False)

    # perform point-in-polygon test for triangles
    if lazy:
        centers_tri = cell_centers_bulk(coords, triangles)
        k_tri = min(k, len(centers_tri))
        neighbours_tri = k_nearest_neighbours(centers_tri, x, k=k_tri)
        nat_tri = _glob_to_nat_tri_bulk_knn_(
            x, ecoords_tri, neighbours_tri
        )  # (nP, kTET, 4)
        pips_tri = __pip_tri_bulk__(nat_tri, tol)  # (nP, kTET)
    else:
        nat_tri = _glob_to_nat_tri_bulk_(x, ecoords_tri)  # (nP, nTET, 4)
        pips_tri = __pip_tri_bulk__(nat_tri, tol)  # (nP, nTET)

    # locate the points that are inside any of the cells
    pip = np.squeeze(np.any(pips_tri, axis=1))  # (nP)
    i_source = np.where(pip)[0]  # (nP_)
    if lazy:
        points_to_tris, points_to_neighbours = _find_first_hits_knn_(
            pips_tri[i_source], neighbours_tri[i_source]
        )
    else:
        points_to_tris, points_to_neighbours = _find_first_hits_(pips_tri[i_source])
    tets_to_cells = np.floor(np.arange(len(triangles)) / len(trimap)).astype(int)
    i_target = tets_to_cells[points_to_tris]  # (nP_)

    # locate the cells that contain the points
    cell_tri_indices = np.tile(np.arange(trimap.shape[0]), len(triangles))[
        points_to_tris
    ]
    nat_tri = nat_tri[i_source]  # (nP_, nTET, 4)
    locations_target = _ntri_to_loc_bulk_(
        lcoords, nat_tri, trimap, cell_tri_indices, points_to_neighbours
    )

    return i_source, i_target, locations_target
