# -*- coding: utf-8 -*-
from os.path import exists
from typing import Union, Any

import meshio

from ..vtkutils import PolyData_to_mesh
from ..config import __haspyvista__

if __haspyvista__:
    import pyvista as pv

    pyVistaLike = Union[pv.PolyData, pv.PointGrid, pv.UnstructuredGrid]
else:
    pyVistaLike = Any


# TODO : read from image file with vtk
def input_to_mesh(*args, **kwargs) -> tuple:
    """
    Reads from several formats and returns the coordinates and the topology
    of a polygonal mesh. Every item in `args` is considered as a seprate
    candidate, but the function is able to handle several inputs at once.

    Parameters
    ----------
    args : Tuple
        Anything that can be converted into a polygonal mesh, that is

            1) a file supported by `meshio`

            2) an instance of `pyvista.PolyData` or `pyvista.UnstructuredGrid`

    Returns
    -------
    List[Tuple[array, Union[array, dict]]]
        A list of tuples, where the second item in each tuple is either a numpy
        array of node indices, or a dictionary of such arrays. The first item
        of the tuples is always a numpy cooridnate array.
    """
    candidate = kwargs.get("__candidate", None)
    if candidate is None:
        res = []
        for arg in args:
            res.append(input_to_mesh(__candidate=arg))
        return res

    coords, topo = None, None

    # read from file with meshio
    # TODO : inp2stl
    if isinstance(candidate, str):
        file_exists = exists(candidate)
        assert file_exists, "The file does not exist on this file system."
        mesh = meshio.read(
            candidate,  # string, os.PathLike, or a buffer/open file
            # file_format="stl",  # optional if filename is a path; inferred from extension
            # see meshio-convert -h for all possible formats
        )
        coords, topo = mesh.points, mesh.cells_dict

    # PyVista
    if coords is None and __haspyvista__:
        if isinstance(candidate, pv.PolyData):
            coords, topo = PolyData_to_mesh(candidate)
        elif isinstance(candidate, pv.UnstructuredGrid):
            coords, topo = candidate.points, candidate.cells_dict

    assert (
        coords is not None
    ), "Failed to read from this input, check the documentation!"
    return coords, topo
