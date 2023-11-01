from ..config import __hask3d__
from ..helpers import plotters

if not __hask3d__:  # pragma: no cover

    def k3dplot(*_, **__) -> None:
        raise ImportError(
            "You need K3D for this. Install it with 'pip install k3d'. "
            "You may also need to restart your kernel and reload the package."
        )

else:
    from typing import Union, Optional

    import k3d

    from ..data import PolyData

    def k3dplot(
        obj: PolyData,
        scene: Optional[Union[k3d.Plot, None]] = None,
        *,
        menu_visibility: Optional[bool] = True,
        **kwargs,
    ) -> k3d.Plot:
        """
        Plots the mesh using 'k3d' as the backend.

        .. warning::
            During this call there is a UserWarning saying 'Given trait value dtype
            "float32" does not match required type "float32"'. Although this is weird,
            plotting seems to be just fine.

        Parameters
        ----------
        scene: object, Optional
            A K3D plot widget to append to. This can also be given as the
            first positional argument. Default is None, in which case it is
            created using a call to :func:`k3d.plot`.
        menu_visibility: bool, Optional
            Whether to show the menu or not. Default is True.
        **kwargs: dict, Optional
            Extra keyword arguments forwarded to :func:`to_k3d`.

        Example
        -------
        Get a compound mesh, add some random data to it and plot it with K3D.

        .. code-block:: python

            # doctest: +SKIP

            from sigmaepsilon.mesh.plotting import k3dplot
            from sigmaepsilon.mesh.examples import compound_mesh
            from k3d.colormaps import matplotlib_color_maps
            import k3d
            import numpy as np
            mesh = compound_mesh()
            cd_L2 = mesh["lines", "L2"].cd
            cd_Q4 = mesh["surfaces", "Q4"].cd
            cd_H8 = mesh["bodies", "H8"].cd
            cd_L2.db["scalars"] = 100 * np.random.rand(len(cd_L2))
            cd_Q4.db["scalars"] = 100 * np.random.rand(len(cd_Q4))
            cd_H8.db["scalars"] = 100 * np.random.rand(len(cd_H8))
            scalars = mesh.pd.pull("scalars")
            cmap = matplotlib_color_maps.Jet
            fig = k3d.plot()
            k3dplot(mesh, fig, scalars=scalars, menu_visibility=False, cmap=cmap)
            fig
        """
        if scene is None:
            scene = k3d.plot(menu_visibility=menu_visibility)
        return obj.to_k3d(scene=scene, **kwargs)


plotters["k3d"] = k3dplot

__all__ = ["k3dplot"]
