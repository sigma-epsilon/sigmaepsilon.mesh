from .cell1d import PolyCell1d


__all__ = ["Line", "QuadraticLine", "NonlinearLine"]


class Line(PolyCell1d):
    """
    Base class for linear 2-noded lines.
    """

    NNODE: int  = 2
    vtkCellType: int  = 3

    
class QuadraticLine(PolyCell1d):
    """
    Base class for quadratic 3-noded lines.
    """

    NNODE: int  = 3
    vtkCellType: int  = 21


class NonlinearLine(PolyCell1d):
    """
    Base class for general nonlinear lines.
    """

    NNODE: int = None
    vtkCellType: int  = None
