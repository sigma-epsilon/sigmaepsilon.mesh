from typing import Union, Callable, Tuple, Optional
from numbers import Number

import numpy as np
from numpy import ndarray

from .typing import PolyCellProtocol
from .data import PointData, PolyData, TriMesh
from .grid import grid
from .cells import H8, H27, TET4, TET10, T3, W6, W18
from .space import CartesianFrame
from .triang import triangulate
from .utils import cell_centers_bulk
from .utils.topology import detach, H8_to_TET4, H27_to_TET10, TET4_to_TET10, W6_to_W18
from .extrude import extrude_T3_TET4, extrude_T3_W6
from .voxelize import voxelize_cylinder


def circular_helix(
    a: Optional[Union[Number, None]] = None,
    b: Optional[Union[Number, None]] = None,
    *,
    slope: Optional[Union[Number, None]] = None,
    pitch: Optional[Union[Number, None]] = None,
) -> Callable[[Number], Tuple[float, float, float]]:
    """
    Returns the function :math:`f(t) = [a \\cdot cos(t), a \\cdot sin(t), b \\cdot t]`,
    which describes a circular helix of radius a and slope a/b (or pitch 2πb).

    Parameters
    ----------
    a: float, Optional
        Radius of the helix. Default is None.
    b: float, Optional
        Slope of the helix. Default is None.
    slope: float, Optional
        Slope of the helix. Default is None.
    pitch: float, Optional
        Pitch of the helix. Default is None.
    """
    if pitch is not None:
        b = b if b is not None else pitch / 2 / np.pi
    if slope is not None:
        a = a if a is not None else slope * b
        b = b if b is not None else slope / a

    def inner(t: Number) -> Tuple[float, float, float]:
        """
        Evaluates :math:`f(t) = [a \\cdot cos(t), a \\cdot sin(t), b \\cdot t]`.
        """
        return a * np.cos(t), a * np.sin(t), b * t

    return inner


def circular_disk(
    nangles: int,
    nradii: int,
    rmin: float,
    rmax: float,
    frame: Optional[Union[CartesianFrame, None]] = None,
) -> TriMesh:
    """
    Returns the triangulation of a circular disk.

    Parameters
    ----------
    nangles: int
        Number of subdivisions in radial direction.
    nradii: int
        Number of subdivisions in circumferential direction.
    rmin: float
        Inner radius. Can be zero.
    rmax: float
        Outer radius.

    Returns
    -------
    TriMesh

    Examples
    --------
    >>> from sigmaepsilon.mesh.recipes import circular_disk
    >>> mesh = circular_disk(120, 60, 5, 25)
    """
    radii = np.linspace(rmin, rmax, nradii)
    angles = np.linspace(0, 2 * np.pi, nangles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], nradii, axis=1)
    angles[:, 1::2] += np.pi / nangles
    x = (radii * np.cos(angles)).flatten()
    y = (radii * np.sin(angles)).flatten()
    nP = len(x)
    points = np.stack([x, y], axis=1)
    *_, triang = triangulate(points=points, backend="mpl")
    # triang = tri.Triangulation(x, y)
    # Mask off unwanted triangles.
    triang.set_mask(
        np.hypot(x[triang.triangles].mean(axis=1), y[triang.triangles].mean(axis=1))
        < rmin
    )
    triangles = triang.get_masked_triangles()
    points = np.stack((triang.x, triang.y, np.zeros(nP)), axis=1)
    points, triangles = detach(points, triangles)
    frame = CartesianFrame(dim=3) if frame is None else frame
    pd = PointData(coords=points, frame=frame)
    cd = T3(topo=triangles, frames=frame)
    return TriMesh(pd, cd)


def cylinder(
    shape: Tuple[Tuple[Number, Number] | Number, Number],
    size: Union[Tuple, float, int],
    *,
    voxelize: bool = False,
    celltype: Optional[Union[PolyCellProtocol, None]] = None,
    frame: Optional[Union[CartesianFrame, None]] = None,
) -> PolyData:
    """
    Returns the coordinates and the topology of cylinder as numpy arrays.

    Parameters
    ----------
    shape: Tuple[Tuple[Number, Number] | Number, Number]
        The parameter is a 2-tuple that describes the dimensions of the
        cylinder. The first element is either a number (radius), or a 2-tuple of
        numbers (inner and outer radii) describing the radius of the cylinder.
        The second element is the height of the cylinder.
    size: Union[tuple, float, int]
        Parameter controlling the resolution of the mesh. Default is None.

        If `voxelize` is ``False``, ``size`` must be a tuple of three
        integers, describing the number of angular, radial, and vertical
        divions in this order.

        If `voxelize` is ``True`` and ``size`` is a ``float``,
        the parameter controls the size of the individual voxels.

        If `voxelize` is ``True`` and ``size`` is an ``int``,
        the parameter controls the size of the individual voxels
        according to :math:`edge \\, length = (r_{ext} - r_{int})/shape`.
    voxelize: bool, Optional
        If ``True``, the cylinder gets voxelized to a collection of H8 cells.
        In this case the size of a voxel can be controlled by specifying a
        flaot or an integer as the second parameter ``size``.
        Default is ``False``.
    celltype
        Specifies the celltype to be used.

    Example
    -------
    >>> from sigmaepsilon.mesh.recipes import cylinder
    >>> n_angles = 30  # number of subdivions along the angular direction
    >>> n_radii = 15  # number of subdivisions along the radial direction
    >>> min_radius = 10 # minimum radius of the cylinder
    >>> max_radius = 25 # maximum radius of the cylinder
    >>> n_z = 20 # number of subdivisions along the z direction
    >>> h = 50 # height of the cylinder
    >>> shape = (min_radius, max_radius), h
    >>> size = n_radii, n_angles, n_z
    >>> mesh = cylinder(shape, size, voxelize=False)

    Returns
    -------
    :class:`~sigmaepsilon.mesh.data.polydata.PolyData`
    """
    if celltype is None:
        celltype = H8 if voxelize else TET4
    etype = None

    if isinstance(size, float) or isinstance(size, int):
        size = [size]

    if voxelize:
        etype = "H8"

    radius, h = shape

    if isinstance(radius, int):
        radius = np.array([0, radius])
    elif not isinstance(radius, ndarray):
        radius = np.array(radius)

    etype = celltype.label if etype is None else etype

    if voxelize:
        if isinstance(size[0], int):
            size_ = (radius[1] - radius[0]) / size[0]
        elif isinstance(size[0], float):
            size_ = size[0]
        coords, topo = voxelize_cylinder(radius=radius, height=h, size=size_)
    else:
        if etype == "TET4":
            min_radius, max_radius = radius
            n_radii, n_angles, n_z = size
            mesh = circular_disk(n_angles, n_radii, min_radius, max_radius)
            points, triangles = mesh.coords(), mesh.topology().to_numpy()
            coords, topo = extrude_T3_TET4(points, triangles, h, n_z)
        else:
            raise NotImplementedError("Celltype not supported!")

    frame = CartesianFrame(dim=3) if frame is None else frame
    pd = PointData(coords=coords, frame=frame)
    cd = celltype(topo=topo, frames=frame)
    return PolyData(pd, cd)


def ribbed_plate(
    lx: float,
    ly: float,
    t: float,
    *,
    wx: float = None,
    wy: float = None,
    hx: float = None,
    hy: float = None,
    ex: float = None,
    ey: float = None,
    lmax: float = None,
    order: int = 1,
    tetrahedralize: bool = False,
) -> PolyData:
    """
    Creates a ribbed plate.

    Parameters
    ----------
    lx: float
        The length of the plate along the X axis.
    ly: float
        The length of the plate along the Y axis.
    t: float
        The thickness of a plate.
    wx: float, Optional
        The width of the ribs running in X direction. Must be defined
        alongside `hx`. Default is None.
    hx: float, Optional
        The height of the ribs running in X direction. Must be defined
        alongside `wx`. Default is None.
    ex: float, Optional
        The eccentricity of the ribs running in X direction.
    wy: float, Optional
        The width of the ribs running in Y direction. Must be defined
        alongside `hy`. Default is None.
    hy: float, Optional
        The height of the ribs running in Y direction. Must be defined
        alongside `wy`. Default is None.
    ey: float, Optional
        The eccentricity of the ribs running in Y direction.
    lmax: float, Optional
        Maximum edge length of the cells in the resulting mesh. Default is None.
    order: int, Optional
        Determines the order of the cells used. Allowed values are 1 and 2. If order is
        1, either H8 hexahedra or TET4 tetrahedra are returned. If order is 2, H27
        hexahedra or TET10 tetrahedra are returned.
    tetrahedralize: bool, Optional
        If True, a mesh of 4-noded tetrahedra is returned. Default is False.

    Example
    -------
    >>> from sigmaepsilon.mesh.recipes import ribbed_plate
    >>> mesh = ribbed_plate(lx=5.0, ly=5.0, t=1.0,
    ...                     wx=1.0, hx=2.0, ex=0.05,
    ...                     wy=1.0, hy=2.0, ey=-0.05)

    Returns
    -------
    :class:`~sigmaepsilon.mesh.data.polydata.PolyData`
    """

    def subdivide(bins, lmax):
        _bins = []
        for i in range(len(bins) - 1):
            a = bins[i]
            b = bins[i + 1]
            if (b - a) > lmax:
                ndiv = int(np.ceil((b - a) / lmax))
            else:
                ndiv = 1
            ldiv = (b - a) / ndiv
            for j in range(ndiv):
                _bins.append(a + j * ldiv)
        _bins.append(bins[-1])
        return np.array(_bins)

    xbins, ybins, zbins = [], [], []
    xbins.extend([-lx / 2, 0, lx / 2])
    ybins.extend([-ly / 2, 0, ly / 2])
    zbins.extend([-t / 2, 0, t / 2])

    if wx is not None and hx is not None:
        ex = 0.0 if ex is None else ex
        ybins.extend([-wx / 2, wx / 2])
        if (ex - hx / 2) < (-t / 2):
            zbins.append(ex - hx / 2)
        if (ex + hx / 2) > (t / 2):
            zbins.append(ex + hx / 2)

    if wy is not None and hy is not None:
        ey = 0.0 if ey is None else ey
        xbins.extend([-wy / 2, wy / 2])
        if (ey - hy / 2) < (-t / 2):
            zbins.append(ey - hy / 2)
        if (ey + hy / 2) > (t / 2):
            zbins.append(ey + hy / 2)

    xbins = np.unique(np.sort(xbins))
    ybins = np.unique(np.sort(ybins))
    zbins = np.unique(np.sort(zbins))
    if isinstance(lmax, float):
        xbins = subdivide(xbins, lmax)
        ybins = subdivide(ybins, lmax)
        zbins = subdivide(zbins, lmax)
    bins = xbins, ybins, zbins

    if order == 1:
        coords, topo = grid(bins=bins, eshape="H8")
    elif order == 2:
        coords, topo = grid(bins=bins, eshape="H27")
    else:
        raise ValueError("'order' must be either 1 or 2")

    centers = cell_centers_bulk(coords, topo)
    mask = (centers[:, 2] > (-t / 2)) & (centers[:, 2] < (t / 2))

    if wx is not None and hx is not None:
        m = (centers[:, 1] > (-wx / 2)) & (centers[:, 1] < (wx / 2))
        m = m & (centers[:, 2] > (ex - hx / 2)) & (centers[:, 2] < (ex + hx / 2))
        mask = mask | m

    if wy is not None and hy is not None:
        m = (centers[:, 0] > (-wy / 2)) & (centers[:, 0] < (wy / 2))
        m = m & (centers[:, 2] > (ey - hy / 2)) & (centers[:, 2] < (ey + hy / 2))
        mask = mask | m

    topo = topo[mask, :]

    if tetrahedralize:
        if order == 1:
            coords, topo = H8_to_TET4(coords, topo)
            celltype = TET4
        else:
            coords, topo = H27_to_TET10(coords, topo)
            celltype = TET10
    else:
        celltype = H8 if order == 1 else H27

    coords, topo = detach(coords, topo)
    frame = CartesianFrame(dim=3)
    pd = PointData(coords=coords, frame=frame)
    cd = celltype(topo=topo, frames=frame)
    return PolyData(pd, cd)


def perforated_cube(
    lx: float,
    ly: float,
    lz: float,
    radius: float,
    *,
    nangles: int = None,
    lmax: float = None,
    order: int = 1,
    prismatic: bool = True,
) -> PolyData:
    """
    Returns a cube of side lengths 'lx', 'ly' and 'lz', with a circular hole
    along the 'z' axis.

    Returns
    -------
    :class:`~sigmaepsilon.mesh.data.polydata.PolyData`
    """
    size = (lx, ly)

    if lmax is not None:
        shape = (max([int(lx / lmax), 4]), max([int(ly / lmax), 4]))
    else:
        shape = (4, 4)
    coords, _ = grid(size=size, shape=shape, eshape=(2, 2), centralize=True)

    if lmax is not None:
        where = np.hypot(coords[:, 0], coords[:, 1]) > (radius + lmax)
    else:
        where = np.hypot(coords[:, 0], coords[:, 1]) > (radius * 1.1)
    coords = coords[where]

    if nangles is None:
        if lmax is not None:
            nangles = max(int(2 * np.pi * radius / lmax), 8)
        else:
            nangles = 16

    angles = np.linspace(0, 2 * np.pi, nangles, endpoint=False)
    x_circle = (radius * np.cos(angles)).flatten()
    y_circle = (radius * np.sin(angles)).flatten()
    circle_coords = np.stack([x_circle, y_circle], axis=1)

    coords = np.vstack([coords[:, :2], circle_coords])

    *_, triobj = triangulate(points=coords, backend="mpl")
    triobj.set_mask(
        np.hypot(
            coords[:, 0][triobj.triangles].mean(axis=1),
            coords[:, 1][triobj.triangles].mean(axis=1),
        )
        < radius
    )
    topo = triobj.get_masked_triangles()
    coords = np.stack((triobj.x, triobj.y, np.zeros(coords.shape[0])), axis=1)
    coords, topo = detach(coords, topo)

    if lmax is not None:
        zres = int(lz / lmax)
    else:
        zres = 4

    if prismatic:
        coords, topo = extrude_T3_W6(coords, topo, h=lz, zres=zres)
    else:
        coords, topo = extrude_T3_TET4(coords, topo, h=lz, zres=zres)

    if order == 1:
        celltype = W6 if prismatic else TET4
    elif order == 2:
        if prismatic:
            coords, topo = W6_to_W18(coords, topo)
            celltype = W18
        else:
            coords, topo = TET4_to_TET10(coords, topo)
            celltype = TET10
    else:
        raise ValueError("'order' must be either 1 or 2")

    frame = CartesianFrame(dim=3)
    pd = PointData(coords=coords, frame=frame)
    cd = celltype(topo=topo, frames=frame)
    return PolyData(pd, cd)


def sphere(
    radius: float,
    n_azimuthal_divisions: int,
    n_polar_divisions: Optional[Union[int, None]] = None,
) -> PolyData:
    """
    Generate a triangulated mesh representing a sphere.

    Parameters
    ----------
    radius : float
        Radius of the sphere.
    n_azimuthal_divisions : int
        Number of divisions for azimuthal angle (longitude).
    n_polar_divisions : int, Optional
        Number of divisions for polar angle (latitude). If not provided
        it is equal to 'n_azimuthal_divisions'.

    Returns
    -------
    :class:`~sigmaepsilon.mesh.data.polydata.PolyData`

    Notes
    -----
    This function generates a triangulated mesh of a sphere using separate
    divisions for azimuthal and polar angles. The resulting mesh represents
    a sphere with given radius and resolution.

    Examples
    --------
    >>> from sigmaepsilon.mesh import PolyData
    >>> from sigmaepsilon.mesh.recipes import sphere
    >>> mesh: PolyData = sphere(1.0, 20, 10)
    """

    if not n_polar_divisions:
        n_polar_divisions = n_azimuthal_divisions

    # Create points (vertices)
    phi = np.linspace(0, np.pi, n_polar_divisions)  # Polar angle
    theta = np.linspace(0, 2 * np.pi, n_azimuthal_divisions)  # Azimuthal angle
    phi, theta = np.meshgrid(phi, theta)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T

    # Create faces (triangles)
    faces = []
    for i in range(n_azimuthal_divisions - 1):
        for j in range(n_polar_divisions - 1):
            p1 = i * n_polar_divisions + j
            p2 = p1 + 1
            p3 = (i + 1) * n_polar_divisions + j
            p4 = p3 + 1

            faces.extend([[p1, p2, p3], [p2, p4, p3]])

    frame = CartesianFrame(dim=3)
    pd = PointData(coords=points, frame=frame)
    cd = T3(topo=np.array(faces))
    mesh = PolyData(pd, cd)
    return mesh
