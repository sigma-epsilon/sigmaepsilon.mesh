from meshio._vtk_common import vtk_to_meshio_type

from .abcakwrapper import ABCMeta_AkWrapper
from .geometry import GeometryProtocol
from ..helpers import vtk_to_celltype, meshio_to_celltype


__all__ = ["ABCMeta_PolyCell", "ABC_PolyCell"]


class ABCMeta_PolyCell(ABCMeta_AkWrapper):
    """
    Meta class for PointData and CellData classes.

    It merges attribute maps with those of the parent classes.
    """

    def __init__(self, name, bases, namespace, *args, **kwargs):
        super().__init__(name, bases, namespace, *args, **kwargs)

    def __new__(metaclass, name, bases, namespace, *args, **kwargs):
        cls = super().__new__(metaclass, name, bases, namespace, *args, **kwargs)

        if namespace.get("Geometry", None):
            if not isinstance(cls.Geometry, GeometryProtocol):
                raise TypeError(
                    f"The attached geometry class {cls.Geometry} of {cls} "
                    "does not implement PolyCellGeometry"
                )
            else:
                # add class to helpers
                vtk_cell_id = getattr(cls.Geometry, "vtk_cell_id", None)
                if isinstance(vtk_cell_id, int):
                    vtk_to_celltype[vtk_cell_id] = cls
                    meshio_to_celltype[vtk_to_meshio_type[vtk_cell_id]] = cls

        return cls


class ABC_PolyCell(metaclass=ABCMeta_PolyCell):
    """
    Helper class that provides a standard way to create an ABC using
    inheritance.
    """

    __slots__ = ()
