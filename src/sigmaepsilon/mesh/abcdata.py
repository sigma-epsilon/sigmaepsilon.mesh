from sigmaepsilon.core.meta import ABCMeta_Weak

from meshio._vtk_common import vtk_to_meshio_type

from .helpers import vtk_to_celltype, meshio_to_celltype


__all__ = ["ABCMeta_MeshData", "ABC_MeshData"]


class ABCMeta_MeshData(ABCMeta_Weak):
    """
    Meta class for PointData and CellData classes.

    It merges attribute maps with those of the parent classes.
    """

    def __init__(self, name, bases, namespace, *args, **kwargs):
        super().__init__(name, bases, namespace, *args, **kwargs)

    def __new__(metaclass, name, bases, namespace, *args, **kwargs):
        cls = super().__new__(metaclass, name, bases, namespace, *args, **kwargs)

        # merge database fields
        _attr_map_ = namespace.get("_attr_map_", {})
        for base in bases:
            _attr_map_.update(base.__dict__.get("_attr_map_", {}))
        cls._attr_map_ = _attr_map_

        # add class to helpers
        vtkCellType = getattr(cls, "vtkCellType", None)
        if isinstance(vtkCellType, int):
            vtk_to_celltype[vtkCellType] = cls
            meshio_to_celltype[vtk_to_meshio_type[vtkCellType]] = cls
        return cls


class ABC_MeshData(metaclass=ABCMeta_MeshData):
    """
    Helper class that provides a standard way to create an ABC using
    inheritance.
    """

    __slots__ = ()
