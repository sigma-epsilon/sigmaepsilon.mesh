from sigmaepsilon.core.meta import ABCMeta_Weak

from meshio._vtk_common import vtk_to_meshio_type

from .geometry import PolyCellGeometry
from ..helpers import vtk_to_celltype, meshio_to_celltype


__all__ = ["ABCMeta_MeshCellData", "ABC_MeshCellData"]


class ABCMeta_MeshCellData(ABCMeta_Weak):
    """
    Meta class for PointData and CellData classes.

    It merges attribute maps with those of the parent classes.
    """

    def __init__(self, name, bases, namespace, *args, **kwargs):
        super().__init__(name, bases, namespace, *args, **kwargs)

    def __new__(metaclass, name, bases, namespace, *args, **kwargs):
        cls = super().__new__(metaclass, name, bases, namespace, *args, **kwargs)

        if namespace.get("Geometry", None):
            if not issubclass(cls.Geometry, PolyCellGeometry):
                raise TypeError(
                    f"The attached geometry class {cls.Geometry} of {cls} "
                    "must be a subclass of PolyCellGeometry"
                )
            else:
                # add class to helpers
                vtk_cell_id = getattr(cls.Geometry, "vtk_cell_id", None)
                if isinstance(vtk_cell_id, int):
                    vtk_to_celltype[vtk_cell_id] = cls
                    meshio_to_celltype[vtk_to_meshio_type[vtk_cell_id]] = cls

        # merge database fields
        _attr_map_ = namespace.get("_attr_map_", {})
        for base in bases:
            _attr_map_.update(base.__dict__.get("_attr_map_", {}))
        cls._attr_map_ = _attr_map_

        return cls


class ABC_MeshCellData(metaclass=ABCMeta_MeshCellData):
    """
    Helper class that provides a standard way to create an ABC using
    inheritance.
    """

    __slots__ = ()
