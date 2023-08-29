from typing import Protocol, Callable, Optional, ClassVar, Iterable, Union, Tuple

from numpy import ndarray

from ...celldata import CellData
from ...topoarray import TopologyArray


class LagrangianPolyCellProtocol(Protocol):
    number_of_nodes: ClassVar[int]
    number_of_spatial_dimensions: ClassVar[int]
    vtk_cell_id: ClassVar[Optional[int]] = None
    meshio_cell_id: ClassVar[Optional[str]] = None
    boundary_class: ClassVar[Optional["LagrangianPolyCellProtocol"]] = None
    shape_function_evaluator: ClassVar[Optional[Callable]] = None
    shape_function_matrix_evaluator: ClassVar[Optional[Callable]] = None
    shape_function_derivative_evaluator: ClassVar[Optional[Callable]] = None
    monomial_evaluator: ClassVar[Optional[Callable]] = None
    
    @classmethod
    def shape_function_values(cls, pcoords: ndarray) -> ndarray:
        """
        Evaluates the shape functions at the specified locations.

        Parameters
        ----------
        pcoords: numpy.ndarray
            Locations of the evaluation points.

        Returns
        -------
        numpy.ndarray
            An array of shape (nP, nNE) where nP and nNE are the number of
            evaluation points and shape functions.
        """
        ...
        
    @classmethod
    def shape_function_derivatives(
        cls, pcoords: Iterable[float], *, jac: ndarray = None, dshp: ndarray = None
    ) -> ndarray:
        """
        Evaluates shape function derivatives wrt. the master element or the local
        coordinate frames of some cells. To control the behaviour, either 'jac' or 'wrt'
        can be provided.

        Parameters
        ----------
        pcoords: Iterable[float]
            Locations of the evaluation points.
        jac: Iterable, Optional
            The jacobian matrix as a float array of shape (nE, nP, nD, nD), evaluated for
            an nP number of points and nP number cells and nD number of spatial dimensions.
            Default is None.
        dshp: numpy.ndarray, Optional
            Shape function derivatives wrt. the master element. Only relevant if 'jac' is
            provided. The purpose of this argument is to avoid repeated evaluation in situations
            where 'dshp' is required on its own and is already at hand when calling this function.
            Default is None, in which case it is calculated automatically.

        Notes
        -----
        Only first derivatives are calculated.

        Returns
        -------
        numpy.ndarray
            An array of shape (nP, nNE, nD), where nP, nNE and nD are
            the number of evaluation points, nodes and spatial dimensions.
            If 'jac' is provided, the result is of shape (nE, nP, nNE, nD),
            where nE is the number of cells in the block.
        """
        ...
        
    @classmethod
    def shape_function_matrix(cls, pcoords: ndarray, *, N: int = None) -> ndarray:
        """
        Evaluates the shape function matrix at the specified locations.

        Parameters
        ----------
        pcoords: numpy.ndarray
            Locations of the evaluation points.
        N: int, Optional
            Number of unknowns per node.

        Returns
        -------
        numpy.ndarray
            An array of shape (nP, nNE, N * nNE) where nP, nDOF and nNE
            are the number of evaluation points, degrees of freedom per node
            and nodes per cell.
        """
        ...
        
    def jacobian_matrix(
        self, db: CellData, *, pcoords: Iterable[float] = None, dshp: ndarray = None, **__
    ) -> ndarray:
        """
        Returns the jacobian matrices of the cells in the block. The evaluation
        of the matrix is governed by the inputs in the following way:
        - if `dshp` is provided, it must be a matrix of shape function derivatives
          evaluated at the desired locations
        - the desired locations are specified through `pcoords`

        Parameters
        ----------
        db: CellData
            The database that feeds the cells.
        pcoords: Iterable[float], Optional
            Locations of the evaluation points.
        dshp: numpy.ndarray, Optional
            3d array of shape function derivatives for the master cell,
            evaluated at some points. The array must have a shape of
            (nG, nNE, nD), where nG, nNE and nD are the number of evaluation
            points, nodes per cell and spatial dimensions.

        Returns
        -------
        numpy.ndarray
            A 4d array of shape (nE, nP, nD, nD), where nE, nP and nD
            are the number of elements, evaluation points and spatial
            dimensions. The number of evaluation points in the output
            is governed by the parameter 'dshp' or 'pcoords'.
        """
        ...
        
    def jacobian(self, db: CellData, *, jac: ndarray = None, **kwargs) -> Union[float, ndarray]:
        """
        Returns the jacobian determinant for one or more cells.

        Parameters
        ----------
        db: CellData
            The database that feeds the cells.
        jac: numpy.ndarray, Optional
            One or more Jacobian matrices. Default is None.
        **kwargs: dict
            Forwarded to :func:`jacobian_matrix` if the jacobian
            is not provided by the parameter 'jac'.

        Returns
        -------
        float or numpy.ndarray
            Value of the Jacobian for one or more cells.
        """
        ...
        
    def coords(self, db: CellData, *args, **kwargs) -> ndarray:
        """
        Returns the coordinates of the cells in the database as a 3d
        numpy array.
        
        Parameters
        ----------
        db: CellData
            The database that feeds the cells.
        """
        ...

    def topology(self, db: CellData) -> TopologyArray:
        """
        Returns the numerical representation of the topology of
        the cells.
        
        Parameters
        ----------
        db: CellData
            The database that feeds the cells.
        """
        ...
        
    def to_simplices(self, db: CellData) -> Tuple[ndarray]:
        """
        Returns the cells of the block, refactorized into simplices.
        
        Parameters
        ----------
        db: CellData
            The database that feeds the cells.
        """