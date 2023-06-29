from .cell import PolyCell1d
from numpy import ndarray

from .utils.utils import jacobian_matrix_bulk_1d, jacobian_det_bulk_1d


__all__ = ["Line", "QuadraticLine", "NonlinearLine"]


class Line(PolyCell1d):
    """
    Base class for all lines.
    """

    NNODE = 2
    vtkCellType = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def jacobian_matrix(self, *args, dshp: ndarray = None, **kwargs):
        """
        Calculates jacobian matrices.

        Parameters
        ----------
        dshp: numpy.ndarray
            Array of shape function derivatives.
        """
        assert dshp is not None
        ecoords = kwargs.get("ecoords", self.local_coordinates())
        return jacobian_matrix_bulk_1d(dshp, ecoords)

    def jacobian(self, *args, jac: ndarray = None, **kwargs):
        """
        Calculates jacobian determinants.

        Parameters
        ----------
        jac: numpy.ndarray
            Array of jacobian matrices derivatives.

        See Also
        --------
        :func:`jacobian_matrix`
        """
        return jacobian_det_bulk_1d(jac)


class QuadraticLine(Line):
    """
    Base class for quadratic 3-noded lines.

    See Also
    --------
    :class:`Line`
    """

    NNODE = 3
    vtkCellType = None


class NonlinearLine(Line):
    """
    Base class for general nonlinear lines.

    See Also
    --------
    :class:`Line`
    """

    NNODE: int = None
    vtkCellType = None
