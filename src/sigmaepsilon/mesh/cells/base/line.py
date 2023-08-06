from .cell1d import PolyCell1d


__all__ = ["Line", "QuadraticLine", "NonlinearLine"]


class Line(PolyCell1d):
    """
    Base class for linear 2-noded lines.
    """

    NNODE = 2
    vtkCellType = 3

    
class QuadraticLine(PolyCell1d):
    """
    Base class for quadratic 3-noded lines.
    """

    NNODE = 3
    vtkCellType = 21


class NonlinearLine(PolyCell1d):
    """
    Base class for general nonlinear lines.
    """

    NNODE: int = None
    vtkCellType = None
