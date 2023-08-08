from typing import Union, MutableMapping, Iterable, Tuple, List, Callable

import numpy as np
from numpy import ndarray
from sympy import Matrix, lambdify

from sigmaepsilon.math import atleast1d, atleast2d, ascont
from sigmaepsilon.math.linalg import ReferenceFrame as FrameLike

from sigmaepsilon.mesh.space import PointCloud, CartesianFrame
from ...celldata import CellData
from ...utils.utils import (
    jacobian_matrix_bulk,
    jacobian_matrix_bulk_1d,
    jacobian_det_bulk_1d,
    points_of_cells,
    pcoords_to_coords,
    global_shape_function_derivatives,
)
from ...utils.cells.utils import (
    _loc_to_glob_bulk_,
)
from ...utils.topology.topo import detach_mesh_bulk, rewire
from ...utils import cell_center, cell_centers_bulk
from ...topoarray import TopologyArray
from ...space import CartesianFrame
from .interpolator import LagrangianCellInterpolator
from ...config import __haspyvista__

MapLike = Union[ndarray, MutableMapping]


class PolyCell(CellData):
    """
    A subclass of :class:`sigmaepsilon.mesh.celldata.CellData` as a base class
    for all kinds of geometrical entities.
    """

    NNODE: int = None  # number of nodes per cell
    NDIM: int = None  # number of spatial dimensions
    vtkCellType: int = None  # vtk Id
    meshioCellType: str = None
    _face_cls_: "PolyCell" = None  # the class of a face
    shpfnc: Callable = None  # evaluator for shape functions
    shpmfnc: Callable = None  # evaluator for shape function matrices
    dshpfnc: Callable = None  # evaluator for shape function derivatives
    monomsfnc: Callable = None  # evaluator for monomials

    def __init__(self, *args, i: ndarray = None, **kwargs):
        if isinstance(i, ndarray):
            kwargs[self._dbkey_id_] = i
        super().__init__(*args, **kwargs)

    @classmethod
    def lcoords(cls) -> ndarray:
        """
        Ought to return local coordinates of the master element.

        Returns
        -------
        numpy.ndarray
        """
        raise NotImplementedError

    @classmethod
    def master_coordinates(cls) -> ndarray:
        """
        Returns the coordinates of the master element.

        Returns
        -------
        numpy.ndarray
        """
        return cls.lcoords()

    @classmethod
    def lcenter(cls) -> ndarray:
        """
        Ought to return the local coordinates of the center of the
        master element.

        Returns
        -------
        numpy.ndarray
        """
        return cell_center(cls.master_coordinates())

    @classmethod
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
        cls, return_symbolic: bool = True, update: bool = True
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
        >>> shp, dshp, shpf, shpmf, dshpf = H8.generate_class_functions()

        Here `shp` and `dshp` are simbolic matrices for shape functions and
        their first derivatives, `shpf`, `shpmf` and `dshpf` are functions for
        fast evaluation of shape function values, the shape function matrix and
        shape function derivatives, respectively.
        """
        nN = cls.NNODE
        nD = cls.NDIM
        nDOF = getattr(cls, "NDOFN", 3)
        locvars, monoms = cls.polybase()
        monoms.pop(0)
        lcoords = cls.lcoords()
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

        def shpmf(p: ndarray, ndof: int = nDOF) -> ndarray:
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
            cls.shpfnc = shpf
            cls.shpmfnc = shpmf
            cls.dshpfnc = dshpf

        if return_symbolic:
            return shp, dshp, shpf, shpmf, dshpf
        else:
            return shpf, shpmf, dshpf

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
            evaluation points and shape functions. If there is only one
            evaluation point, the returned array is one dimensional.
        """
        pcoords = np.array(pcoords)
        if cls.shpfnc is None:
            cls.generate_class_functions(update=True)
        if cls.NDIM == 3:
            if len(pcoords.shape) == 1:
                pcoords = atleast2d(pcoords, front=True)
                return cls.shpfnc(pcoords).astype(float)
        return cls.shpfnc(pcoords).astype(float)

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
        nDOFN = getattr(cls, "NDOFN", N) if N is None else N
        pcoords = np.array(pcoords)
        if cls.shpmfnc is None:
            cls.generate_class_functions(update=True)
        if cls.NDIM == 3:
            if len(pcoords.shape) == 1:
                pcoords = atleast2d(pcoords, front=True)
                if nDOFN:
                    return cls.shpmfnc(pcoords, nDOFN).astype(float)
                else:
                    return cls.shpmfnc(pcoords).astype(float)
        if nDOFN:
            return cls.shpmfnc(pcoords, nDOFN).astype(float)
        else:
            return cls.shpmfnc(pcoords).astype(float)

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
        """
        if jac is None:
            pcoords = np.array(pcoords) if pcoords is not None else cls.lcoords()
            if cls.dshpfnc is None:
                cls.generate_class_functions(update=True)
            if cls.NDIM == 3:
                if len(pcoords.shape) == 1:
                    pcoords = atleast2d(pcoords, front=True)
                    return cls.dshpfnc(pcoords).astype(float)
            return cls.dshpfnc(pcoords).astype(float)
        else:
            pcoords = np.array(pcoords) if pcoords is not None else cls.lcoords()
            if dshp is None:
                dshp = cls.shape_function_derivatives(pcoords)
            return global_shape_function_derivatives(dshp, jac)

    @classmethod
    def interpolator(cls, x: Iterable = None) -> LagrangianCellInterpolator:
        """
        Returns a callable object that can be used to interpolate over
        nodal values of one or more cells.
        
        Parameters
        ----------
        x: Iterable, Optional
            The locations of known data. It can be fed into the returned interpolator
            function directly, but since the operation involves the inversion of a matrix
            related to these locations, it is a good idea to pre calculate it if you want
            to reuse the interpolator with the same source coordinates.
            
        Returns
        -------
        :class:`~sigmaepsilon.mesh.cells.base.interpolator.LagrangianCellInterpolator`
            A callable interpolator class. Refer to its documentation for more examples.
        
        Notes
        -----
        If the number of source coorindates does not match the number of nodes (and hence
        the number of shape functions) of the master element of the class, the interpolation
        is gonna be under or overdetermined and the operation involves the calculation of a 
        generalized inverse.
        
        See also
        --------
        :class:`~sigmaepsilon.mesh.cells.LagrangianCellInterpolator`
        
        Examples
        --------
        Let assume that we know some data at some locations:
        >>> source_data = [1, 2, 3, 4]
        >>> source_location = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
    
        We want to extrapolate from this information to the location
        >>> target_location = [[-2, -2], [2, -2], [2, 2], [-2, 2]]
        
        We have provided four points and four data values. If we want an exact extrapolation,
        we use 4-noded quadrilaterals:
        
        >>> from sigmaepsilon.mesh import Q4
        >>> interpolator = Q4.interpolator()
        >>> target_data = interpolator(source=source_location, values=source_data, target=target_location)
        
        Here we provided 3 inputs to the interpolator. If we want to reuse the interpolator
        with the same source locations, it is best to provide them when creating the interpolator.
        This saves some time for repeated evaluations.
        
        >>> from sigmaepsilon.mesh import Q4
        >>> interpolator = Q4.interpolator(source_location)
        >>> target_data = interpolator(values=source_data, target=target_location)
        """
        return LagrangianCellInterpolator(cls, x)
        
    def jacobian_matrix(
        self, *, pcoords: Iterable[float] = None, dshp: ndarray = None, **__
    ) -> ndarray:
        """
        Returns the jacobian matrices of the cells in the block. The evaluation
        of the matrix is governed by the inputs in the following way:
        - if `dshp` is provided, it must be a matrix of shape function derivatives
          evaluated at the desired locations
        - the desired locations are specified through `pcoords`

        Parameters
        ----------
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
        ecoords = self.local_coordinates()
        if dshp is None:
            x = np.array(pcoords) if pcoords is not None else self.lcoords()
            dshp = self.shape_function_derivatives(x)
        if self.NDIM == 1:
            return jacobian_matrix_bulk_1d(dshp, ecoords)
        else:
            return jacobian_matrix_bulk(dshp, ecoords)

    def jacobian(self, *, jac: ndarray = None, **kwargs) -> Union[float, ndarray]:
        """
        Returns the jacobian determinant for one or more cells.

        Parameters
        ----------
        jac: numpy.ndarray, Optional
            One or more Jacobian matrices. Default is None.
        **kwargs: dict
            Forwarded to :func:`jacobian_matrix` if the jacobian
            is not provided by the parameter 'jac'.

        Returns
        -------
        float or numpy.ndarray
            Value of the Jacobian for one or more cells.

        See Also
        --------
        :func:`jacobian_matrix`
        """
        if jac is None:
            jac = self.jacobian_matrix(**kwargs)
        if self.NDIM == 1:
            return jacobian_det_bulk_1d(jac)
        else:
            return np.linalg.det(jac)

    def flip(self) -> "PolyCell":
        """
        Reverse the order of nodes of the topology.
        """
        topo = self.topology().to_numpy()
        self.nodes = np.flip(topo, axis=1)
        return self

    def measures(self, *args, **kwargs) -> ndarray:
        """Ought to return measures for each cell in the database."""
        raise NotImplementedError

    def measure(self, *args, **kwargs) -> float:
        """Ought to return the net measure for the cells in the
        database as a group."""
        return np.sum(self.measures(*args, **kwargs))

    def area(self, *args, **kwargs) -> float:
        """Returns the total area of the cells in the database. Only for 2d entities."""
        return np.sum(self.areas(*args, **kwargs))

    def areas(self, *args, **kwargs) -> ndarray:
        """Ought to return the areas of the individuall cells in the database."""
        raise NotImplementedError

    def volume(self, *args, **kwargs) -> float:
        """Returns the volume of the cells in the database."""
        return np.sum(self.volumes(*args, **kwargs))

    def volumes(self, *args, **kwargs) -> ndarray:
        """Ought to return the volumes of the individual cells in the database."""
        raise NotImplementedError

    def extract_surface(self, detach: bool = False):
        """Extracts the surface of the mesh. Only for 3d meshes."""
        raise NotImplementedError

    def source_points(self) -> PointCloud:
        """
        Returns the hosting pointcloud.
        """
        return self.container.source().points()

    def source_coords(self) -> ndarray:
        """
        Returns the coordinates of the hosting pointcloud.
        """
        if self.pointdata is not None:
            coords = self.pointdata.x
        else:
            coords = self.container.source().coords()
        return coords

    def source_frame(self) -> FrameLike:
        """
        Returns the frame of the hosting pointcloud.
        """
        return self.container.source().frame

    def points_of_cells(
        self,
        *,
        points: Union[float, Iterable] = None,
        cells: Union[int, Iterable] = None,
        target: Union[str, CartesianFrame] = "global",
    ) -> ndarray:
        """
        Returns the points of selected cells as a NumPy array.
        """
        if cells is not None:
            cells = atleast1d(cells)
            conds = np.isin(cells, self.id)
            cells = atleast1d(cells[conds])
            assert len(cells) > 0, "Length of cells is zero!"
        else:
            cells = np.s_[:]

        if isinstance(target, str):
            assert target.lower() in ["global", "g"]
        else:
            raise NotImplementedError

        coords = self.source_coords()
        topo = self.topology().to_numpy()[cells]
        ecoords = points_of_cells(coords, topo, centralize=False)

        if points is None:
            return ecoords
        else:
            points = np.array(points)

        shp = self.shape_function_values(points)
        if len(shp) == 3:  # variable metric cells
            shp = shp if len(shp) == 2 else shp[cells]

        return pcoords_to_coords(points, ecoords, shp)  # (nE, nP, nD)

    def local_coordinates(self, *, target: CartesianFrame = None) -> ndarray:
        """
        Returns local coordinates of the cells as a 3d float
        numpy array.

        Parameters
        ----------
        target: CartesianFrame, Optional
            A target frame. If provided, coordinates are returned in
            this frame, otherwise they are returned in the local frames
            of the cells. Default is None.
        """
        if isinstance(target, CartesianFrame):
            frames = target.show()
        else:
            frames = self.frames
        topo = self.topology().to_numpy()
        if self.pointdata is not None:
            coords = self.pointdata.x
        else:
            coords = self.container.source().coords()
        return points_of_cells(coords, topo, local_axes=frames, centralize=True)

    def coords(self, *args, **kwargs) -> ndarray:
        """
        Returns the coordinates of the cells in the database as a 3d
        numpy array.
        """
        return self.points_of_cells(*args, **kwargs)

    def topology(self) -> TopologyArray:
        """
        Returns the numerical representation of the topology of
        the cells.
        """
        key = self._dbkey_nodes_
        if key in self.fields:
            return TopologyArray(self.nodes)
        else:
            return None

    def rewire(self, imap: MapLike = None, invert: bool = False) -> "PolyCell":
        """
        Rewires the topology of the block according to the mapping
        described by the argument `imap`. The mapping of the j-th node
        of the i-th cell happens the following way:

        topology_new[i, j] = imap[topology_old[i, j]]

        The object is returned for continuation.

        Parameters
        ----------
        imap: MapLike
            Mapping from old to new node indices (global to local).
        invert: bool, Optional
            If `True` the argument `imap` describes a local to global
            mapping and an inversion takes place. In this case,
            `imap` must be a `numpy` array. Default is False.
        """
        if imap is None:
            imap = self.source().pointdata.id
        topo = self.topology().to_array().astype(int)
        topo = rewire(topo, imap, invert=invert).astype(int)
        self._wrapped[self._dbkey_nodes_] = topo
        return self

    def glob_to_loc(self, x: Union[Iterable, ndarray]) -> ndarray:
        """
        Returns the local coordinates of the input points for each
        cell in the block. The input 'x' can describe a single (1d array),
        or several positions at once (2d array).

        Notes
        -----
        This function is useful when detecting if two bodies touch each other or not,
        and if they do, where.

        Parameters
        ----------
        x: Iterable or numpy.ndarray
            A single point in 3d space as an 1d array, or a collection of points
            as a 2d array.

        Returns
        -------
        numpy.ndarray
            A NumPy array of shape (nE, nP, nD), where nP is the number of points in 'x',
            nE is the number of cells in the block and nD is the number of spatial dimensions.
        """
        raise NotImplementedError

    def loc_to_glob(self, x: Union[Iterable, ndarray]) -> ndarray:
        """
        Returns the global coordinates of the input points for each
        cell in the block. The input 'x' can describe a single (1d array),
        or several local positions at once (2d array).

        Notes
        -----
        This function is useful when detecting if two bodies touch each other or not,
        and if they do, where.

        Parameters
        ----------
        x: Iterable or numpy.ndarray
            A single point as an 1d array, or a collection of points
            as a 2d array.

        Returns
        -------
        numpy.ndarray
            A NumPy array of shape (nE, nP, nD), where nP is the number of points in 'x',
            nE is the number of cells in the block and nD is the number of spatial dimensions.
        """
        x = atleast2d(x, front=True)
        shp = self.shape_function_values(x)
        ecoords = self.points_of_cells()
        return _loc_to_glob_bulk_(shp.T, ecoords)

    def pip(
        self,
        x: Union[Iterable, ndarray],
        tol: float = 1e-12,
        lazy: bool = True,
        k: int = 4,
    ) -> Union[bool, ndarray]:
        """
        Returns an 1d boolean integer array that tells if the points specified by 'x'
        are included in any of the cells in the block.

        Parameters
        ----------
        x: Iterable or numpy.ndarray
            The coordinates of the points that we want to investigate.
        tol: float, Optional
            Floating point tolerance for detecting boundaries. Default is 1e-12.
        lazy: bool, Optional
            If False, the ckeck is performed for all cells in the block. If True,
            it is used in combination with parameter 'k' and the check is only performed
            for the k nearest neighbours of the input points. Default is True.
        k: int, Optional
            The number of neighbours for the case when 'lazy' is true. Default is 4.

        Returns
        -------
        bool or numpy.ndarray
            A single or NumPy array of booleans for every input point.
        """
        raise NotImplementedError

    def locate(
        self,
        x: Union[Iterable, ndarray],
        lazy: bool = True,
        tol: float = 1e-12,
        k: int = 4,
    ) -> Tuple[ndarray]:
        """
        Locates a set of points inside the cells of the block.

        Parameters
        ----------
        x: Iterable or numpy.ndarray
            The coordinates of the points that we want to investigate.
        tol: float, Optional
            Floating point tolerance for detecting boundaries. Default is 1e-12.
        lazy: bool, Optional
            If False, the ckeck is performed for all cells in the block. If True,
            it is used in combination with parameter 'k' and the check is only performed
            for the k nearest neighbours of the input points. Default is True.
        k: int, Optional
            The number of neighbours for the case when 'lazy' is true. Default is 4.

        Returns
        -------
        numpy.ndarray
            The indices of 'x' that are inside a cell of the block.
        numpy.ndarray
            The block-local indices of the cells that include the points with
            the returned indices.
        numpy.ndarray
            The parametric coordinates of the located points inside the including cells.
        """
        raise NotImplementedError

    def to_simplices(self) -> Tuple[ndarray]:
        """
        Returns the cells of the block, refactorized into simplices.
        """
        NDIM = self.__class__.NDIM
        if NDIM == 1:
            return self.to_simplices()
        elif NDIM == 2:
            return self.to_triangles()
        elif NDIM == 3:
            return self.to_tetrahedra()

    def centers(self, target: FrameLike = None) -> ndarray:
        """Returns the centers of the cells of the block."""
        coords = self.source_coords()
        t = self.topology().to_numpy()
        centers = cell_centers_bulk(coords, t)
        if target:
            pc = PointCloud(centers, frame=self.source_frame())
            centers = pc.show(target)
        return centers

    def unique_indices(self) -> ndarray:
        """
        Returns the indices of the points involved in the cells of the block.
        """
        return np.unique(self.topology())

    def points_involved(self) -> PointCloud:
        """Returns the points involved in the cells of the block."""
        return self.source_points()[self.unique_indices()]

    def detach_points_cells(self) -> Tuple[ndarray]:
        """
        Returns the detached coordinate and topology array of the block.
        """
        coords = self.container.source().coords()
        topo = self.topology().to_numpy()
        return detach_mesh_bulk(coords, topo)

    def _rotate_(self, *args, **kwargs):
        # this is triggered upon transformations performed on the hosting pointcloud
        if self.has_frames:
            source_frame = self.container.source().frame
            new_frames = (
                CartesianFrame(self.frames, assume_cartesian=True)
                .rotate(*args, **kwargs)
                .show(source_frame)
            )
            self.frames = new_frames
