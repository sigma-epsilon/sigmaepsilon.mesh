import numpy as np
from numbers import Number

from sectionproperties.pre.pre import DEFAULT_MATERIAL, Material
from sectionproperties.pre.geometry import Geometry
from sectionproperties.pre.library.steel_sections import circular_hollow_section as CHS
from sectionproperties.pre.library.steel_sections import (
    rectangular_hollow_section as RHS,
)
from sectionproperties.pre.library.primitive_sections import rectangular_section as RS
from sectionproperties.pre.library.steel_sections import i_section
from sectionproperties.pre.library.steel_sections import tapered_flange_i_section as TFI
from sectionproperties.pre.library.steel_sections import channel_section as PFC
from sectionproperties.pre.library.steel_sections import tapered_flange_channel as TFC
from sectionproperties.pre.library.steel_sections import tee_section
from sectionproperties.analysis.section import Section

from sigmaepsilon.core.wrapping import Wrapper
from sigmaepsilon.core.kwargtools import getallfromkwargs
from sigmaepsilon.mesh.utils import centralize
from sigmaepsilon.mesh.data import TriMesh, PolyData
from sigmaepsilon.mesh.utils.topology import T6_to_T3, detach_mesh_bulk

from ..cells import T3
from ..data import PointData
from ..space import CartesianFrame
from ..utils import coords_to_3d
from ..topoarray import TopologyArray

__all__ = ["generate_mesh", "get_section", "LineSection"]


def generate_mesh(
    geometry: Geometry, *, l_max: float = None, a_max: float = None, n_max: int = None
) -> Geometry:
    """
    Calculates a float describing the maximum mesh element area to be used
    for the finite-element mesh of the section.

    Parameters
    ----------
    geometry: :class:`sectionproperties.pre.geometry.Geometry`
        Describes the shape of the section, i.e. 'I', 'H', 'RHS', etc.
    l_max: float, Optional
        Maximum element edge length. Default is None.
    a_max: float, Optional
        Maximum element area. Default is None.
    n_max: int, Optional
        Maximum number of elements. Default is None.

    Note
    ----
    The value of the mesh size is derived from the assumption of a perfectly
    regular mesh of equilateral triangles.

    Returns
    -------
    :class:`sectionproperties.pre.geometry.Geometry`
        The geometry object provided with a finite element mesh.
    """
    area = geometry.calculate_area()
    mesh_sizes_max = []

    if isinstance(l_max, Number):
        mesh_sizes_max.append(l_max**2 * np.sqrt(3) / 4)

    if isinstance(a_max, Number):
        mesh_sizes_max.append(a_max)

    if isinstance(n_max, int):
        mesh_sizes_max.append(area / n_max)

    mesh_size_max = None
    if len(mesh_sizes_max) > 0:
        mesh_size_max = min(mesh_sizes_max)
    else:
        raise ValueError(
            "Invalid input, the maximum mesh size can't be tedermined!"
        )  # pragma: no cover

    geometry.create_mesh(mesh_sizes=[mesh_size_max])
    return geometry


def get_section(
    shape: str,
    *,
    mesh_params: dict | None = None,
    material: Material | None = None,
    **section_params,
) -> Section:
    """
    Returns a :class:`sectionproperties.analysis.section.Section` instance.

    The parameters in `section_params` are forwarded to the appropriate constructor of the
    `sectionproperties` library. For the available parameters, see their documentation:

        https://sectionproperties.readthedocs.io/en/latest/rst/section_library.html

    Parameters
    ----------
    shape: str
        Describes the shape of the section. The currently supported section types
        are 'I', 'CHS', 'RHS', 'TFI', 'PFC', 'TFC', 'T'.
    mesh_params: dict, Optional
        A dictionary of parameters controlling mesh generation. Default is None.
        For the possible keys and values see :func:`generate_mesh`, to which
        these parameters are forwarded to. Default is None.
    material: :class:`~sectionproperties.pre.pre.Material`, Optional
        The material of the section. If not specified, a default material is
        used. Default is None.
    **section_params: dict
        Parameters required for a given section. See the documentation of
        the `sectionproperties` library for more details. The parameters
        are only required, if the first argument is a string.

    Note
    ----
    Specification of a material is only necessary, if stresses are to be
    calculated. If the reason of creating the section is the calculation of
    geometrical properties of a section, the choice of material is irrelevant.

    Returns
    -------
    :class:`sectionproperties.analysis.section.Section`
        An object representing a cross-section of a beam.

    Examples
    --------
    >>> from sigmaepsilon.mesh.domains.section import get_section
    >>> mesh_params = dict(n_max=500)
    >>> section = get_section('CHS', d=1.0, t=0.1, n=64, mesh_params=mesh_params)
    """

    material = DEFAULT_MATERIAL if material is None else material
    geom, ms = None, None
    if shape == "CHS":
        geom = CHS(
            d=section_params["d"],
            t=section_params["t"],
            n=section_params.get("n", 64),
            material=material,
        )
        ms = section_params["t"]
    elif shape == "RS":
        keys = ["d", "b"]
        params = getallfromkwargs(keys, **section_params)
        params_dict = {k: v for k, v in zip(keys, params)}
        geom = RS(material=material, **params_dict)
        ms = min(params)
    elif shape == "RHS":
        keys = ["d", "b", "t", "r_out", "n_r"]
        params = getallfromkwargs(keys, **section_params)
        params_dict = {k: v for k, v in zip(keys, params)}
        geom = RHS(material=material, **params_dict)
        ms = section_params["t"]
    elif shape == "I":
        keys = ["d", "b", "t_f", "t_w", "r", "n_r"]
        params = getallfromkwargs(keys, **section_params)
        params_dict = {k: v for k, v in zip(keys, params)}
        geom = i_section(material=material, **params_dict)
        ms = min(section_params["t_f"], section_params["t_w"])
    elif shape == "TFI":
        keys = ["d", "b", "t_f", "t_w", "r_r", "r_f", "alpha", "n_r"]
        params = getallfromkwargs(keys, **section_params)
        params_dict = {k: v for k, v in zip(keys, params)}
        geom = TFI(material=material, **params_dict)
        ms = min(section_params["t_f"], section_params["t_w"])
    elif shape == "PFC":
        keys = ["d", "b", "t_f", "t_w", "r", "n_r"]
        params = getallfromkwargs(keys, **section_params)
        params_dict = {k: v for k, v in zip(keys, params)}
        geom = PFC(material=material, **params_dict)
        ms = min(section_params["t_f"], section_params["t_w"])
    elif shape == "TFC":
        keys = ["d", "b", "t_f", "t_w", "r_r", "r_f", "alpha", "n_r"]
        params = getallfromkwargs(keys, **section_params)
        params_dict = {k: v for k, v in zip(keys, params)}
        geom = TFC(material=material, **params_dict)
        ms = min(section_params["t_f"], section_params["t_w"])
    elif shape == "T":
        keys = ["d", "b", "t_f", "t_w", "r", "n_r"]
        params = getallfromkwargs(keys, **section_params)
        params_dict = {k: v for k, v in zip(keys, params)}
        geom = tee_section(material=material, **params_dict)
        ms = min(section_params["t_f"], section_params["t_w"])
    else:
        raise NotImplementedError(
            "Section type <{}> is not yet implemented :(".format(shape)
        )

    if geom is not None:
        if mesh_params is None:
            assert ms is not None, "Invalid input!"
            mesh_params = dict(l_max=ms)
        assert isinstance(mesh_params, dict)
        geom = generate_mesh(geom, **mesh_params)
        return Section(geom)
    raise RuntimeError("Unable to get section.")  # pragma: no cover


class LineSection(Wrapper):
    """
    Wraps an instance of `sectionproperties.analysis.section.Section` and
    adds a little here and there to make some of the functionality more
    accessible.

    Parameters
    ----------
    *args: tuple, Optional
        The first parameter can be a string referring to a section type, or an
        instance of :class:`sectionproperties.analysis.section.Section`. In the
        former case, parameters of the section can be provided as keyword arguments,
        which are then forwarded to :func:`get_section`, see its documentation
        for further details.
    **kwargs: dict, Optional
        The parameters of a section as keyword arguments, only if the first
        positional argument is a string (see above).
    mesh_params: dict, Optional
        A dictionary controlling the density of the mesh of the section.
        Default is None.
    material: :class:`sectionproperties.pre.pre.Material`, Optional
        The material of the section. If not specified, a default material is
        used. Default is None.
    wrap: :class:`sectionproperties.analysis.section.Section`
        A section object to be wrapped. It can also be provided as the
        first positional argument.
    Notes
    -----
    The implementation here only covers homogeneous cross sections. If you want
    to define an inhomogeneous section, it must be explicity given either as
    the first position argument, or with the keyword argument `wrap`.

    Examples
    --------
    >>> from sigmaepsilon.mesh.domains.section import LineSection, get_section
    >>> section = LineSection(get_section('CHS', d=1.0, t=0.1, n=64))

    or simply provide the shape as the first argument and everything
    else with keyword arguments:

    >>> section = LineSection('CHS', d=1.0, t=0.1, n=64)

    Example
    -------
    .. plot::
        :include-source: True

        Plot a section with Matplotlib using 3-noded triangles:

        import matplotlib.pyplot as plt
        from sigmaepsilon.mesh.plotting.mpl import triplot_mpl_mesh
        section = LineSection('CHS', d=1.0, t=0.3, n=32,
                              mesh_params=dict(n_max=20))
        triobj = section.trimesh(order=1).to_triobj()
        fig, ax = plt.subplots(figsize=(4, 2))
        triplot_mpl_mesh(triobj, fig=fig, ax=ax, lw=0.1)
    """

    def __init__(
        self,
        *args,
        wrap=None,
        shape=None,
        mesh_params=None,
        material: Material = None,
        **kwargs,
    ):
        if len(args) > 0:
            try:
                if isinstance(args[0], str):
                    wrap = get_section(
                        args[0], mesh_params=mesh_params, material=material, **kwargs
                    )
                else:
                    if shape is None:
                        if isinstance(args[0], Section):
                            wrap = args[0]
                    else:
                        wrap = get_section(
                            shape, mesh_params=mesh_params, material=material, **kwargs
                        )
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"Unable to create section: {e}")

        super().__init__(*args, wrap=wrap, **kwargs)
        self.props = None

    def coords(self) -> np.ndarray:
        """
        Returns centralized vertex coordinates of the supporting
        point cloud as a numpy array.
        """
        return centralize(np.array(self.mesh["vertices"]))

    def topology(self) -> TopologyArray:
        """
        Returns vertex indices of T6 triangles as a numpy array.
        """
        return TopologyArray(np.array(self.mesh["triangles"].tolist()))

    def trimesh(self, subdivide: bool = False, order: int = 1, **kwargs) -> TriMesh:
        """
        Returns the mesh of the section as a collection of T3 triangles.
        Keyword arguments are forwarded to the constructor of
        :class:`~sigmaepsilon.mesh.tri.trimesh.TriMesh`.

        Parameters
        ----------
        order: boolean, Optional
            Order of the tetrahedra. Order 1 means linear, order 2 quadratic. Default is 1.
        subdivide: boolean, Optional
            Controls how the T6 triangles are transformed into T3 triangles,
            if the argument 'T6' is False. If True, the T6 triangles
            are subdivided into 4 T3 triangles. If False, T3 triangles are
            formed by the corners of T6 triangles only, and all the remaining
            nodes are neglected.

        See Also
        --------
        :class:`~sigmaepsilon.mesh.trimesh.TriMesh`

        Examples
        --------
        >>> from sigmaepsilon.mesh.domains import LineSection
        >>> section = LineSection(get_section('CHS', d=1.0, t=0.1, n=64))
        >>> trimesh = section.trimesh()
        """
        triangles = self.topology()
        assert (
            not triangles.is_jagged()
        ), "Only homogeneous topologies are supported at the moment."
        points, triangles = coords_to_3d(self.coords()), triangles.to_numpy()
        if order == 1:
            if subdivide:
                path = np.array([[0, 5, 4], [5, 1, 3], [3, 2, 4], [5, 3, 4]], dtype=int)
                points, triangles = T6_to_T3(points, triangles, path=path)
            else:
                points, triangles = detach_mesh_bulk(points, triangles[:, :3])
        else:  # pragma: no cover
            raise NotImplementedError("Higher order elements are not yet supported!")

        frame = kwargs.get("frame", CartesianFrame(dim=3))
        pd = PointData(coords=points, frame=frame)
        cd = T3(topo=triangles, frames=frame)
        return TriMesh(pd, cd)

    def extrude(
        self,
        *,
        length: float,
        frame: CartesianFrame,
        N: int,
        **__,
    ) -> PolyData:
        """
        Creates a 3d tetragonal mesh from the section.

        Parameters
        ----------
        length: float
            Length of the beam.
        N: int
            Number of subdivisions along the length of the beam.
        frame: CartesianFrame
            An object instance representing an orthonormal coordinate frame.

        Returns
        -------
        :class:`~sigmaepsilon.mesh.data.polydata.PolyData`
        """
        return self.trimesh(frame=frame).extrude(h=length, N=N)

    def calculate_geometric_properties(self, *args, **kwargs):
        return self._wrapped.calculate_geometric_properties(*args, **kwargs)

    def calculate_warping_properties(self, *args, **kwargs):
        return self._wrapped.calculate_warping_properties(*args, **kwargs)

    @property
    def A(self) -> float:
        """
        Returns the cross-sectional area.
        """
        return self.section_props.area

    @property
    def Ix(self) -> float:
        """
        Returns the second moment of inertia around 'x'.
        """
        return self.Iy + self.Iz

    @property
    def Iy(self) -> float:
        """
        Returns the second moment of inertia around 'y'.
        """
        return self.section_props.ixx_c

    @property
    def Iz(self) -> float:
        """
        Returns the second moment of inertia around 'z'.
        """
        return self.section_props.iyy_c

    @property
    def geometric_properties(self) -> dict:
        """
        Returns the geometric properties of the section.
        """
        return {"A": self.A, "Ix": self.Ix, "Iy": self.Iy, "Iz": self.Iz}

    @property
    def section_properties(self) -> dict:
        """
        Returns all properties of the section.
        """
        return self.props

    def calculate_section_properties(self) -> dict:
        """
        Retruns a dictionary containing the properties of the section.
        """
        self.calculate_geometric_properties()
        section_properties = self.geometric_properties
        self.calculate_warping_properties()
        self.props = section_properties
        return section_properties

    def get_section_properties(self) -> dict:
        """
        Returns all properties of the section.
        """
        if self.props is not None:
            return self.props
        else:
            return self.calculate_section_properties()
