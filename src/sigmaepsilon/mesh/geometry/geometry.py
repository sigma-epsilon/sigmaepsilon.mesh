from typing import (
    Callable,
    Optional,
    ClassVar,
    Iterable,
    Tuple,
    List,
    Union,
)
from abc import abstractmethod

import numpy as np
from numpy import ndarray
from sympy import Matrix, lambdify, symbols

from sigmaepsilon.core.meta import ABCMeta_Weak
from sigmaepsilon.math import atleast1d, atleast2d, ascont
from sigmaepsilon.math.utils import to_range_1d

from ..typing import GeometryProtocol
from ..utils import cell_center, cell_center_2d
from ..utils.utils import global_shape_function_derivatives
from ..cellapproximator import LagrangianCellApproximator
from ..triang import triangulate


__all__ = [
    "PolyCellGeometry1d",
    "PolyCellGeometry2d",
    "PolyCellGeometry3d",
]


class ABC(metaclass=ABCMeta_Weak):
    """
    Helper class that provides a standard way to create an ABC using
    inheritance.
    """

    __slots__ = ()


class PolyCellGeometry(ABC):
    """
    Base class for classes that implement the geometry protocol.

    See Also
    --------
    :class:`~sigmaepsilon.mesh.typing.geometry.GeometryProtocol`
    """

    number_of_nodes: ClassVar[int]
    number_of_spatial_dimensions: ClassVar[int]
    number_of_nodal_variables = 1
    vtk_cell_id: ClassVar[Optional[int]] = None
    meshio_cell_id: ClassVar[Optional[str]] = None
    boundary_class: ClassVar[Optional["PolyCellGeometry"]] = None
    shape_function_evaluator: ClassVar[Optional[Callable]] = None
    shape_function_matrix_evaluator: ClassVar[Optional[Callable]] = None
    shape_function_derivative_evaluator: ClassVar[Optional[Callable]] = None
    monomial_evaluator: ClassVar[Optional[Callable]] = None
    quadrature: ClassVar[Optional[dict]] = None

    @classmethod
    def generate_class(
        cls, base: Optional[Union[GeometryProtocol, None]] = None, **kwargs
    ) -> GeometryProtocol:
        """
        A factory function that returns a custom 1d class.

        Parameters
        ----------
        base: GeometryProtocol, Optional
            A base class that implements the GeometryProtocol. If not
            provided, the class serves as a base.
        **kwargs: dict, Optional
            A dictionary of class attributes and their values.

        Notes
        -----
        During generation, the generated class only inherits some properties,
        while others are set to default values to avoid inconsistent attributes.

        Example
        -------
        Define a custom 1d cell with 4 nodes:

        >>> from sigmaepsilon.mesh.geometry import PolyCellGeometry1d
        >>> CustomClass = PolyCellGeometry1d.generate_class(number_of_nodes=4)

        This is equivalent to:

        >>> class CustomClass(PolyCellGeometry1d):
        ...     number_of_nodes = 4
        """
        base = cls if not base else base

        class CustomClass(base):
            vtk_cell_id = None
            meshio_cell_id = None
            boundary_class = None
            shape_function_evaluator = None
            shape_function_matrix_evaluator = None
            shape_function_derivative_evaluator = None
            monomial_evaluator = None
            quadrature = None

        for key, value in kwargs.items():
            setattr(CustomClass, key, value)

        return CustomClass

    @classmethod
    @abstractmethod
    def master_coordinates(cls) -> ndarray:
        """
        Returns the coordinates of the master element.

        Returns
        -------
        numpy.ndarray
        """
        ...

    def master_center(cls) -> ndarray:
        """
        Returns the coordinates of the master element.

        Returns
        -------
        numpy.ndarray
        """
        return cell_center(cls.master_coordinates())

    @classmethod
    def shape_function_values(
        cls, x: Union[float, Iterable[float]], *, rng: Iterable = None
    ) -> ndarray:
        """
        Evaluates the shape functions at the specified locations.

        Parameters
        ----------
        x: Union[float, Iterable[float]]
            Locations of the evaluation points in the master domain.
        rng: Iterable, Optional
            The range in which the locations ought to be understood, only for 1d
            cells. Typically [0, 1] or [-1, 1]. Default is [0, 1].

        Returns
        -------
        numpy.ndarray
            An array of shape (nP, nNE) where nP and nNE are the number of
            evaluation points and shape functions. If there is only one
            evaluation point, the returned array is one dimensional.
        """
        if cls.number_of_spatial_dimensions == 1:
            rng = np.array([-1, 1]) if rng is None else np.array(rng)
            x = atleast1d(np.array(x))
            x = to_range_1d(x, source=rng, target=[-1, 1])
        else:
            x = np.array(x)

        if cls.shape_function_evaluator is None:
            cls.generate_class_functions(update=True)

        if cls.number_of_spatial_dimensions == 3:
            if len(x.shape) == 1:
                x = atleast2d(x, front=True)
                return cls.shape_function_evaluator(x).astype(float)

        return cls.shape_function_evaluator(x).astype(float)

    @classmethod
    def shape_function_derivatives(
        cls,
        x: Union[float, Iterable[float]],
        *,
        jac: ndarray = None,
        dshp: ndarray = None,
        rng: Iterable = None,
    ) -> ndarray:
        """
        Evaluates shape function derivatives wrt. the master element or the local
        coordinate frames of some cells. To control the behaviour, either 'jac' or 'wrt'
        can be provided.

        Parameters
        ----------
        x: Union[float, Iterable[float]]
            Locations of the evaluation points.
        jac: Iterable, Optional
            The jacobian matrix as a float array of shape (nE, nP, nD, nD), evaluated for
            an nP number of points and nP number cells and nD number of spatial dimensions.
            Default is None.
        rng: Iterable, Optional
            The range in which the locations ought to be understood, only for 1d
            cells. Typically [0, 1] or [-1, 1]. Default is [0, 1].
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
        if x is not None:
            if cls.number_of_spatial_dimensions == 1:
                rng = np.array([-1, 1]) if rng is None else np.array(rng)
                x = atleast1d(np.array(x))
                x = to_range_1d(x, source=rng, target=[-1, 1])
            else:
                x = np.array(x)

        if jac is None:
            x = np.array(x) if x is not None else cls.master_coordinates()

            if cls.shape_function_derivative_evaluator is None:
                cls.generate_class_functions(update=True)

            if cls.number_of_spatial_dimensions == 3:
                if len(x.shape) == 1:
                    x = atleast2d(x, front=True)
                    return cls.shape_function_derivative_evaluator(x).astype(float)

            return cls.shape_function_derivative_evaluator(x).astype(float)
        else:
            x = np.array(x) if x is not None else cls.master_coordinates()

            if dshp is None:
                dshp = cls.shape_function_derivatives(x)

            return global_shape_function_derivatives(dshp, jac)

    @classmethod
    def shape_function_matrix(
        cls, x: Union[float, Iterable[float]], *, rng: Iterable = None, N: int = None
    ) -> ndarray:
        """
        Evaluates the shape function matrix at the specified locations.

        Parameters
        ----------
        x: Union[float, Iterable[float]]
            Locations of the evaluation points.
        rng: Iterable, Optional
            The range in which the locations ought to be understood, only for 1d
            cells. Typically [0, 1] or [-1, 1]. Default is [0, 1].
        N: int, Optional
            Number of unknowns per node.

        Returns
        -------
        numpy.ndarray
            An array of shape (nP, nNE, N * nNE) where nP, nDOF and nNE
            are the number of evaluation points, degrees of freedom per node
            and nodes per cell.
        """
        if cls.number_of_spatial_dimensions == 1:
            rng = np.array([-1, 1]) if rng is None else np.array(rng)
            x = atleast1d(np.array(x))
            x = to_range_1d(x, source=rng, target=[-1, 1])
        else:
            x = np.array(x)

        if cls.shape_function_matrix_evaluator is None:
            cls.generate_class_functions(update=True)

        if cls.number_of_spatial_dimensions == 3:
            if len(x.shape) == 1:
                x = atleast2d(x, front=True)
                if N:
                    return cls.shape_function_matrix_evaluator(x, N).astype(float)
                else:
                    return cls.shape_function_matrix_evaluator(x).astype(float)

        if N:
            return cls.shape_function_matrix_evaluator(x, N).astype(float)
        else:
            return cls.shape_function_matrix_evaluator(x).astype(float)

    def polybase(cls) -> Tuple[List]:
        """
        Ought to retrun the polynomial base of the master element.

        Returns
        -------
        list
            A list of SymPy symbols.
        list
            A list of monomials.
        """
        raise NotImplementedError

    @classmethod
    def generate_class_functions(
        cls,
        return_symbolic: Optional[bool] = True,
        update: Optional[bool] = True,
    ) -> Tuple:
        """
        Generates functions to evaulate shape functions, their derivatives
        and the shape function matrices using SymPy. For this to work, the
        'polybase' and 'lcoords' class methods must be implemented.

        Parameters
        ----------
        return_symbolic: bool, Optional
            If True, the function returns symbolic expressions of shape functions
            and their derivatives. Default is True.
        update: bool, Optional
            If True, class methods are updated with the generated versions.
            Default is True.

        Notes
        -----
        Some cells are equipped with functions to evaluate shape functions a-priori,
        other classes rely on symbolic generation of these functions. In the latter case,
        this function is automatically invoked runtime, there is no need to manually
        trigger it.

        Example
        -------
        >>> from sigmaepsilon.mesh.cells import H8
        >>> shp, dshp, shpf, shpmf, dshpf = H8.Geometry.generate_class_functions()

        Here `shp` and `dshp` are simbolic matrices for shape functions and
        their first derivatives, `shpf`, `shpmf` and `dshpf` are functions for
        fast evaluation of shape function values, the shape function matrix and
        shape function derivatives, respectively.
        """
        nN = cls.number_of_nodes
        nD = cls.number_of_spatial_dimensions
        nX = cls.number_of_nodal_variables
        locvars, monoms = cls.polybase()
        monoms.pop(0)
        lcoords = cls.master_coordinates()
        if nD == 1:
            lcoords = np.reshape(lcoords, (nN, 1))

        def subs(lpos):
            return {v: lpos[i] for i, v in enumerate(locvars)}

        def mval(lpos):
            return [m.evalf(subs=subs(lpos)) for m in monoms]

        M = np.ones((nN, nN), dtype=float)
        M[:, 1:] = np.vstack([mval(loc) for loc in lcoords])
        coeffs = np.linalg.inv(M)
        monoms.insert(0, 1)
        shp = Matrix([np.dot(coeffs[:, i], monoms) for i in range(nN)])
        dshp = Matrix([[f.diff(m) for m in locvars] for f in shp])
        _shpf = lambdify([locvars], shp[:, 0].T, "numpy")
        _dshpf = lambdify([locvars], dshp, "numpy")

        def shpf(p: ndarray) -> ndarray:
            """
            Evaluates the shape functions at multiple points in the
            master domain.
            """
            p = atleast2d(p, back=True)
            r = np.stack([_shpf(p[i])[0] for i in range(len(p))])
            return ascont(r)

        def shpmf(p: ndarray, ndof: int = nX) -> ndarray:
            """
            Evaluates the shape function matrix at multiple points
            in the master domain.
            """
            p = atleast2d(p, back=True)
            nP = p.shape[0]
            eye = np.eye(ndof, dtype=float)
            shp = shpf(p)
            res = np.zeros((nP, ndof, nN * ndof), dtype=float)
            for iP in range(nP):
                for i in range(nN):
                    res[iP, :, i * ndof : (i + 1) * ndof] = eye * shp[iP, i]
            return ascont(res)

        def dshpf(p: ndarray) -> ndarray:
            """
            Evaluates the shape function derivatives at multiple points
            in the master domain.
            """
            p = atleast2d(p, back=True)
            r = np.stack([_dshpf(p[i]) for i in range(len(p))])
            return ascont(r)

        if update:
            cls.shape_function_evaluator = shpf
            cls.shape_function_matrix_evaluator = shpmf
            cls.shape_function_derivative_evaluator = dshpf

        if return_symbolic:
            return shp, dshp, shpf, shpmf, dshpf
        else:
            return shpf, shpmf, dshpf

    @classmethod
    def approximator(cls, x: Iterable = None) -> LagrangianCellApproximator:
        """
        Returns a callable object that can be used to approximate over
        nodal values of one or more cells.

        Parameters
        ----------
        x: Iterable, Optional
            The locations of known data. It can be fed into the returned approximator
            function directly, but since the operation involves the inversion of a matrix
            related to these locations, it is a good idea to pre calculate it if you want
            to reuse the approximator with the same source coordinates.

        Returns
        -------
        :class:`~sigmaepsilon.mesh.cellapproximator.LagrangianCellApproximator`
            A callable approximator class. Refer to its documentation for more examples.

        Notes
        -----
        If the number of source coorindates does not match the number of nodes (and hence
        the number of shape functions) of the master element of the class, the interpolation
        is gonna be under or overdetermined and the operation involves the calculation of a
        generalized inverse.

        See also
        --------
        :class:`~sigmaepsilon.mesh.cellapproximator.LagrangianCellApproximator`

        Examples
        --------
        Let assume that we know some data at some locations:
        >>> source_data = [1, 2, 3, 4]
        >>> source_location = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]

        We want to extrapolate from this information to the location
        >>> target_location = [[-2, -2], [2, -2], [2, 2], [-2, 2]]

        We have provided four points and four data values. If we want an exact extrapolation,
        we use 4-noded quadrilaterals:

        >>> from sigmaepsilon.mesh.cells import Q4
        >>> approximator = Q4.Geometry.approximator()
        >>> target_data = approximator(source=source_location, values=source_data, target=target_location)

        Here we provided 3 inputs to the approximator. If we want to reuse the approximator
        with the same source locations, it is best to provide them when creating the approximator.
        This saves some time for repeated evaluations.

        >>> from sigmaepsilon.mesh.cells import Q4
        >>> approximator = Q4.Geometry.approximator(source_location)
        >>> target_data = approximator(values=source_data, target=target_location)
        """
        return LagrangianCellApproximator(cls, x)


class PolyCellGeometry1d(PolyCellGeometry):
    number_of_spatial_dimensions = 1

    @classmethod
    def polybase(cls) -> Tuple[List]:
        """
        Retruns the polynomial base of the master element.

        Returns
        -------
        list
            A list of SymPy symbols.
        list
            A list of monomials.
        """
        if not isinstance(cls.number_of_nodes, int):
            raise ValueError(
                "Attribute 'number_of_nodes' of the cell must be set to a positive integer"
            )
        locvars = r = symbols("r", real=True)
        monoms = [r**i for i in range(cls.number_of_nodes)]
        return [locvars], monoms

    @classmethod
    def master_coordinates(cls) -> ndarray:
        """
        Returns local coordinates of the cell.

        Returns
        -------
        numpy.ndarray
        """
        if not isinstance(cls.number_of_nodes, int):
            raise ValueError(
                "Attribute 'number_of_nodes' of the cell must be set to a positive integer"
            )
        return np.linspace(-1.0, 1.0, cls.number_of_nodes)

    @classmethod
    def master_center(cls) -> ndarray:
        """
        Returns the local coordinates of the center of the cell.

        Returns
        -------
        numpy.ndarray
        """
        return np.array([0.0])


class PolyCellGeometry2d(PolyCellGeometry):
    number_of_spatial_dimensions = 2

    @classmethod
    def master_center(cls) -> ndarray:
        """
        Ought to return the local coordinates of the center of the
        master element.

        Returns
        -------
        numpy.ndarray
        """
        return cell_center_2d(cls.master_coordinates())

    @classmethod
    def trimap(cls) -> Iterable:
        """
        Returns a mapper to transform topology and other data to
        a collection of triangular simplices.
        """
        _, t, _ = triangulate(points=cls.master_coordinates())
        return t


class PolyCellGeometry3d(PolyCellGeometry):
    number_of_spatial_dimensions = 3

    @classmethod
    def tetmap(cls) -> Iterable:
        """
        Returns a mapper to transform topology and other data to
        a collection of tetrahedral simplices.
        """
        raise NotImplementedError
