# -*- coding: utf-8 -*-
import numpy as np

from .config import __hasvtk__

if __hasvtk__:
    import vtk
    from vtk.util.numpy_support import (
        numpy_to_vtk as np2vtk,
        numpy_to_vtkIdTypeArray as np2vtkId,
    )
    from vtk.numpy_interface import dataset_adapter as dsa


def mesh_to_vtkdata(coords, topo, deepcopy=True):
    if not __hasvtk__:
        raise ImportError
    # points
    vtkpoints = vtk.vtkPoints()
    vtkpoints.SetData(np2vtk(coords, deep=deepcopy))
    # cells
    topo_vtk = np.concatenate(
        (np.ones((topo.shape[0], 1), dtype=int) * topo.shape[1], topo), axis=1
    ).ravel()
    vtkcells = vtk.vtkCellArray()
    vtkcells.SetNumberOfCells(topo.shape[0])
    vtkcells.SetCells(topo.shape[0], np2vtkId(topo_vtk, deep=deepcopy))
    # return points and cells
    return vtkpoints, vtkcells


def mesh_to_UnstructuredGrid(coords, topo, vtkCellType, deepcopy=True):
    vtkpoints, vtkcells = mesh_to_vtkdata(
        coords.astype(np.float64), topo.astype(np.int64), deepcopy
    )
    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(vtkpoints)
    ugrid.SetCells(vtkCellType, vtkcells)
    return ugrid


def mesh_to_PolyData(coords, topo, deepcopy=True):
    vtkpoints, vtkcells = mesh_to_vtkdata(coords, topo, deepcopy)
    vtkPolyData = vtk.vtkPolyData()
    vtkPolyData.SetPoints(vtkpoints)
    vtkPolyData.SetPolys(vtkcells)
    vtkPolyData.Modified()
    return vtkPolyData


def PolyData_to_mesh(pd, v=0):
    if v == 0:
        coords = pd.points.astype(float)
        ugrid = pd.cast_to_unstructured_grid()
        topo = ugrid.cells_dict
    elif v == 1:
        with dsa.WrapDataObject(pd) as wrapper:
            coords = wrapper.Points
            topo = wrapper.Polygons
    elif v == 2:
        coords = vtk_to_numpy(pd.GetPoints().GetData())
        topo = vtk_to_numpy(pd.GetPolys().GetData())
    return coords, topo
