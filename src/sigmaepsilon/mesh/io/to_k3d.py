from ..config import __hask3d__, __hasmatplotlib__
from ..helpers import exporters

if __hask3d__ and __hasmatplotlib__:
    from copy import copy

    import k3d
    import numpy as np
    from numpy import ndarray
    import matplotlib as mpl

    from sigmaepsilon.math import minmax

    from ..data import PolyData
    from ..utils.topology import detach_mesh_data_bulk, detach_mesh_bulk

    def to_k3d(
        obj: PolyData,
        *,
        scene: object = None,
        deep: bool = True,
        config_key: str = None,
        menu_visibility: bool = True,
        cmap: list = None,
        show_edges: bool = True,
        scalars: ndarray = None,
    ) -> object:
        """
        Returns the mesh as a k3d mesh object.

        :: warning:
            Calling this method raises a UserWarning inside the `traittypes`
            package saying "Given trait value dtype 'float32' does not match
            required type 'float32'." However, plotting seems to be fine.

        Returns
        -------
        object
            A K3D Plot Widget, which is a result of a call to `k3d.plot`.
        """
        if scene is None:
            scene = k3d.plot(menu_visibility=menu_visibility)

        source = obj.source()
        coords = source.coords()

        if isinstance(scalars, ndarray):
            color_range = minmax(scalars)
            color_range = [scalars.min() - 1, scalars.max() + 1]

        k3dparams = dict(wireframe=False)
        if config_key is None:
            config_key = obj.__class__._k3d_config_key_

        for b in obj.cellblocks(inclusive=True, deep=deep):
            NDIM = b.celltype.Geometry.number_of_spatial_dimensions
            params = copy(k3dparams)
            config = b._get_config_(config_key)
            params.update(config)

            if "color" in params:
                if isinstance(params["color"], str):
                    hexstr = mpl.colors.to_hex(params["color"])
                    params["color"] = int("0x" + hexstr[1:], 16)

            if cmap is not None:
                params["color_map"] = cmap

            if NDIM == 1:
                topo = b.cd.topology().to_numpy()

                if isinstance(scalars, ndarray):
                    c, d, t = detach_mesh_data_bulk(coords, topo, scalars)
                    params["attribute"] = d
                    params["color_range"] = color_range
                    params["indices_type"] = "segment"
                else:
                    c, t = detach_mesh_bulk(coords, topo)
                    params["indices_type"] = "segment"

                c = c.astype(np.float32)
                t = t.astype(np.uint32)
                scene += k3d.lines(c, t, **params)
            elif NDIM == 2:
                topo = b.cd.to_triangles()

                if isinstance(scalars, ndarray):
                    c, d, t = detach_mesh_data_bulk(coords, topo, scalars)
                    params["attribute"] = d
                    params["color_range"] = color_range
                else:
                    c, t = detach_mesh_bulk(coords, topo)

                c = c.astype(np.float32)
                t = t.astype(np.uint32)

                if "side" in params:
                    if params["side"].lower() == "both":
                        params["side"] = "front"
                        scene += k3d.mesh(c, t, **params)
                        params["side"] = "back"
                        scene += k3d.mesh(c, t, **params)
                    else:
                        scene += k3d.mesh(c, t, **params)
                else:
                    scene += k3d.mesh(c, t, **params)

                if show_edges:
                    scene += k3d.mesh(c, t, wireframe=True, color=0)
            elif NDIM == 3:
                topo = b.surface().topology().to_numpy()

                if isinstance(scalars, ndarray):
                    c, d, t = detach_mesh_data_bulk(coords, topo, scalars)
                    params["attribute"] = d
                    params["color_range"] = color_range
                else:
                    c, t = detach_mesh_bulk(coords, topo)

                c = c.astype(np.float32)
                t = t.astype(np.uint32)
                scene += k3d.mesh(c, t, **params)

                if show_edges:
                    scene += k3d.mesh(c, t, wireframe=True, color=0)

        return scene

else: # pragma: no cover

    def to_k3d(*_, **__):
        raise ImportError(
            "You need both K3D and Matplotlib for this. Install it with 'pip install k3d matplotlib'. "
            "You may also need to restart your kernel and reload the package."
        )


exporters["k3d"] = to_k3d

__all__ = ["to_k3d"]
