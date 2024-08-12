from ...config import __hasmatplotlib__

if not __hasmatplotlib__:  # pragma: no cover

    def triplot_mpl_mesh(*_, **__):
        raise ImportError(
            "You need Matplotlib for this. Install it with 'pip install matplotlib'. "
            "You may also need to restart your kernel and reload the package."
        )

    def triplot_mpl_hinton(*_, **__):
        raise ImportError(
            "You need Matplotlib for this. Install it with 'pip install matplotlib'. "
            "You may also need to restart your kernel and reload the package."
        )

    def triplot_mpl_data(*_, **__):
        raise ImportError(
            "You need Matplotlib for this. Install it with 'pip install matplotlib'. "
            "You may also need to restart your kernel and reload the package."
        )

else:
    from typing import Any, Union, Optional, Iterable

    import numpy as np
    from numpy import ndarray

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.tri as mpltri
    from matplotlib.figure import Figure, Axes

    from sigmaepsilon.math.utils import minmax
    from sigmaepsilon.mesh.triang import triobj_to_mpl, get_triobj_data, triangulate
    from sigmaepsilon.mesh.utils.tri import offset_tri
    from sigmaepsilon.mesh.utils import cells_coords, explode_mesh_data_bulk

    from .utils import decorate_mpl_ax, triplotter, TriPatchCollection

    @triplotter
    def triplot_mpl_hinton(
        triobj: Any,
        data: ndarray,
        *_,
        fig: Optional[Union[Figure, None]] = None,
        ax: Optional[Union[Axes, Iterable[Axes], None]] = None,
        lw: Optional[float] = 0.5,
        fcolor: Optional[str] = "b",
        ecolor: Optional[str] = "k",
        title: Optional[Union[str, None]] = None,
        suptitle: Optional[Union[str, None]] = None,
        label: Optional[Union[str, None]] = None,
        **kwargs,
    ) -> Any:
        """
        Creates a hinton plot of triangles. The idea is from this example from
        `Matplotlib`:

        https://matplotlib.org/stable/gallery/specialty_plots/hinton_demo.html

        Parameters
        ----------
        triobj: 'TriangulationLike'
            This is either a tuple of mesh data (coordinates and topology)
            or a triangulation object understood by :func:~`sigmaepsilon.mesh.triang.triangulate`.
        data: numpy.ndarray, Optional
            Some data to plot as an 1d NumPy array.
        fig: matplotlib.figure.Figure, Optional
            A `matplotlib` figure to plot on. Default is `None`.
        ax: matplotlib.axes.Axes or Iterable[matplotlib.axes.Axes], Optional.
            A `matplotlib` axis or more if data is 2 dimensional. Default is `None`.
        ecolor: str, Optional
            The color of the edges of the original triangles, before scaling. Default is 'k'.
        fcolor: str, Optional
            The color of the triangles. Default is 'b'.
        lw: Number, Optional
            The linewidth. Default is 0.5.
        title: str, Optional
            Title of the plot. See `matplotlib` for further details. Default is `None`.
        suptitle: str, Optional
            The subtitle of the plot. See `matplotlib` for further details. Default is `None`.
        label: str or Iterable[str], Optional
            The label of the axis. Default is `None`.
        **kwargs**: dict, Optional
            The extra keyword arguments are forwarded to
            :func:`~sigmaepsilon.mesh.plotting.mpl.utils.decorate_mpl_ax`.

        See also
        --------
        :func:`~sigmaepsilon.mesh.plotting.mpl.utils.decorate_mpl_ax`

        Example
        -------
        .. plot::
            :include-source: True

            import numpy as np
            from sigmaepsilon.mesh import grid
            from sigmaepsilon.mesh.utils.topology.tr import Q4_to_T3
            from sigmaepsilon.mesh import triangulate
            from sigmaepsilon.mesh.plotting.mpl.triplot import triplot_mpl_hinton

            gridparams = {
                "size": (1200, 600),
                "shape": (30, 15),
                "eshape": (2, 2),
            }
            coordsQ4, topoQ4 = grid(**gridparams)
            points, triangles = Q4_to_T3(coordsQ4, topoQ4, path="grid")
            triobj = triangulate(points=points[:, :2], triangles=triangles)[-1]

            data = np.random.rand(len(triangles))
            triplot_mpl_hinton(triobj, data=data)
        """
        axobj = None
        tri = triobj_to_mpl(triobj)
        points, triangles = get_triobj_data(tri, trim2d=True)
        cellcoords = offset_tri(points, triangles, data)
        patch = TriPatchCollection(cellcoords, fc=fcolor, ec=ecolor, lw=lw)
        axobj = ax.add_collection(patch)
        decorate_mpl_ax(
            fig=fig,
            ax=ax,
            points=points,
            title=title,
            suptitle=suptitle,
            label=label,
            **kwargs,
        )
        return axobj

    @triplotter
    def triplot_mpl_mesh(
        triobj: Any,
        *_,
        fig: Optional[Union[Figure, None]] = None,
        ax: Optional[Union[Axes, None]] = None,
        lw: Optional[float] = 0.5,
        marker: Optional[str] = "b-",
        zorder: Optional[Union[int, None]] = None,
        fcolor: Optional[Union[str, None]] = None,
        ecolor: Optional[str] = "k",
        title: Optional[Union[str, None]] = None,
        suptitle: Optional[Union[str, None]] = None,
        label: Optional[Union[str, None]] = None,
        **kwargs,
    ) -> Any:
        """
        Plots the mesh of a triangulation.

        Parameters
        ----------
        triobj: 'TriangulationLike'
            This is either a tuple of mesh data (coordinates and topology)
            or a triangulation object understood by :func:~`sigmaepsilon.mesh.triang.triangulate`.
        fig: matplotlib.figure.Figure, Optional
            A `matplotlib` figure to plot on. Default is `None`.
        ax: matplotlib.axes.Axes or Iterable[matplotlib.axes.Axes], Optional.
            A `matplotlib` axis or more if data is 2 dimensional. Default is `None`.
        ecolor: str, Optional
            The color of the edges of the original triangles, before scaling. Default is 'k'.
        fcolor: str, Optional
            The color of the triangles. Default is 'b'.
        lw: Number, Optional
            The linewidth. Default is 0.5.
        zorder: int, Optional
            The zorder of the plot. See `matplotlib` for further details. Default is `None`.
        title: str, Optional
            Title of the plot. See `matplotlib` for further details. Default is `None`.
        suptitle: str, Optional
            The subtitle of the plot. See `matplotlib` for further details. Default is `None`.
        label: str or Iterable[str], Optional
            The label of the axis. Default is `None`.
        **kwargs**: dict, Optional
            The extra keyword arguments are forwarded to
            :func:`~sigmaepsilon.mesh.plotting.mpl.utils.decorate_mpl_ax`.

        See also
        --------
        :func:`~sigmaepsilon.mesh.plotting.mpl.utils.decorate_mpl_ax`

        Example
        -------
        .. plot::
            :include-source: True

            from sigmaepsilon.mesh import grid
            from sigmaepsilon.mesh.utils.topology.tr import Q4_to_T3
            from sigmaepsilon.mesh import triangulate
            from sigmaepsilon.mesh.plotting.mpl.triplot import triplot_mpl_mesh

            gridparams = {
                "size": (1200, 600),
                "shape": (30, 15),
                "eshape": (2, 2),
            }
            coordsQ4, topoQ4 = grid(**gridparams)
            points, triangles = Q4_to_T3(coordsQ4, topoQ4, path="grid")
            triobj = triangulate(points=points[:, :2], triangles=triangles)[-1]

            triplot_mpl_mesh(triobj)
        """
        axobj = None
        tri = triobj_to_mpl(triobj)
        points, triangles = get_triobj_data(tri, trim2d=True)

        if fcolor is None:
            if zorder is not None:
                axobj = ax.triplot(tri, marker, lw=lw, zorder=zorder, **kwargs)
            else:
                axobj = ax.triplot(tri, marker, lw=lw, **kwargs)
        else:
            cellcoords = cells_coords(points, triangles)
            axobj = TriPatchCollection(cellcoords, fc=fcolor, ec=ecolor, lw=lw)
            ax.add_collection(axobj)
        decorate_mpl_ax(
            fig=fig,
            ax=ax,
            points=points,
            title=title,
            suptitle=suptitle,
            label=label,
            **kwargs,
        )
        return axobj

    @triplotter
    def triplot_mpl_data(
        triobj: Any,
        data: ndarray,
        *_,
        fig: Optional[Union[Figure, None]] = None,
        ax: Optional[Union[Axes, Iterable[Axes], None]] = None,
        cmap: Optional[str] = "jet",
        ecolor: Optional[str] = "k",
        lw: Optional[float] = 0.1,
        title: Optional[Union[str, None]] = None,
        suptitle: Optional[Union[str, None]] = None,
        label: Optional[Union[str, None]] = None,
        nlevels: Optional[Union[int, None]] = None,
        refine: Optional[bool] = False,
        refiner: Optional[Union[Any, None]] = None,
        colorbar: Optional[bool] = True,
        subdiv: Optional[int] = 3,
        cbpad: Optional[str] = "2%",
        cbsize: Optional[str] = "5%",
        cbpos: Optional[str] = "right",
        draw_contours: Optional[bool] = True,
        shading: Optional[str] = "flat",
        contour_colors: Optional[str] = "auto",
        **kwargs,
    ) -> Any:
        """
        Convenience function to plot data over triangulations using `matplotlib`. Depending on
        the arguments, the function calls `matplotlib.pyplot.tricontourf` (optionally followed
        by `matplotlib.pyplot.tricontour`) or `matplotlib.pyplot.tripcolor`.

        Parameters
        ----------
        triobj: 'TriangulationLike'
            This is either a tuple of mesh data (coordinates and topology)
            or a triangulation object understood by :func:~`sigmaepsilon.mesh.triang.triangulate`.
        data: numpy.ndarray, Optional
            Some data to plot as an 1d or 2d NumPy array.
        fig: matplotlib.figure.Figure, Optional
            A `matplotlib` figure to plot on. Default is `None`.
        ax: matplotlib.axes.Axes or Iterable[matplotlib.axes.Axes], Optional.
            A `matplotlib` axis or more if data is 2 dimensional. Default is `None`.
        cmap: str, Optional
            The name of a colormap. The default is 'jet'.
        ecolor: str, Optional
            The color of the edges of the triangles. This is only used if data
            is provided over the cells. Default is 'k'.
        lw: Number, Optional
            The linewidth. This is only used if data is provided over the cells. Default is 0.1.
        title: str, Optional
            Title of the plot. See `matplotlib` for further details. Default is `None`.
        suptitle: str, Optional
            The subtitle of the plot. See `matplotlib` for further details. Default is `None`.
        label: str or Iterable[str], Optional
            The label of the axis or more labels if data is 2 dimensional and there are more axes.
            See `matplotlib` for further details. Default is `None`.
        nlevels: int, Optional
            Number of levels on the colorbar, only if `colorbar` is `True`. Default is `None`, in which case
            the colorbar has a continuous distribution. See the examples for the details.
        refine: bool, Optional
            Wether to refine the values. Default is `False`.
        refiner: Any, Optional
            A valid `matplotlib` refiner, only if `refine` is True. If not specified, a `UniformTriRefiner`
            is used. Default is `None`.
        subdiv: int, Optional
            Number of subdivisions for the refiner, only if `refine` is `True`. Default is 3.
        cbpad: str, Optional
            The padding of the colorbar. Default is "2%".
        cbpos: str, Optional
            The position of the colorbar. Default is "right".
        colorbar: bool, Optional
            Wether to put a colorbar or not. Default is `False`.
        draw_contours: bool, Optional
            Wether to draw contour levels or not. Only if data is provided over the cells and `nlevels` is
            also specified. Default is `True`.
        contour_colors: str, Optional
            The color of the contourlines, only if `draw_contours` is `True`. Default is 'auto', which means
            the same as is used for the plot.
        shading: str, Optional
            Shading for `matplotlib.pyplot.tripcolor`, for the case if `nlevels` is None. Default is 'flat'.
        **kwargs**: dict, Optional
            The extra keyword arguments are forwarded to
            :func:`~sigmaepsilon.mesh.plotting.mpl.utils.decorate_mpl_ax`.

        See also
        --------
        :func:`~sigmaepsilon.mesh.plotting.mpl.utils.decorate_mpl_ax`

        Examples
        --------
        .. plot::
            :include-source: True

            import numpy as np
            from sigmaepsilon.mesh import grid
            from sigmaepsilon.mesh.utils.topology.tr import Q4_to_T3
            from sigmaepsilon.mesh import triangulate
            from sigmaepsilon.mesh.plotting.mpl.triplot import triplot_mpl_data

            gridparams = {
                "size": (1200, 600),
                "shape": (30, 15),
                "eshape": (2, 2),
            }
            coordsQ4, topoQ4 = grid(**gridparams)
            points, triangles = Q4_to_T3(coordsQ4, topoQ4, path="grid")
            triobj = triangulate(points=points[:, :2], triangles=triangles)[-1]

            # Data defined over the triangles
            data = np.random.rand(len(triangles))
            triplot_mpl_data(triobj, data=data)

            # Data defined over the points
            data = np.random.rand(len(points))
            triplot_mpl_data(
                triobj, data=data, cmap="jet", nlevels=10, refine=True, draw_contours=True
            )
        """

        axobj = None
        tri = triobj_to_mpl(triobj)
        points, triangles = get_triobj_data(tri, trim2d=True)
        dmin, dmax = minmax(data)

        if refiner is not None:
            refine = True

        nData = len(data)
        if nData == len(triangles):
            nD = len(data.shape)
            if nD == 1:
                axobj = ax.tripcolor(
                    tri, facecolors=data, cmap=cmap, edgecolors=ecolor, lw=lw
                )
            elif nD == 2 and data.shape[1] == 3:
                points, triangles, data = explode_mesh_data_bulk(
                    points, triangles, data
                )
                triobj = triangulate(points=points, triangles=triangles)[-1]
                tri = triobj_to_mpl(triobj)
                axobj = ax.tripcolor(tri, data, cmap=cmap, edgecolors=ecolor, lw=lw)
        elif nData == len(points):
            if refine:
                if refiner is None:
                    refiner = mpltri.UniformTriRefiner(triobj)
                tri, data = refiner.refine_field(data, subdiv=subdiv)

            if isinstance(nlevels, int):
                levels = np.linspace(dmin, dmax, nlevels + 1)
                axobj = ax.tricontourf(tri, data, levels=levels, cmap=cmap)
                if draw_contours:
                    contour_colors = (
                        None if contour_colors == "auto" else contour_colors
                    )
                    ax.tricontour(tri, data, levels=levels, colors=contour_colors)
            else:
                axobj = ax.tripcolor(tri, data, cmap=cmap, shading=shading)
                dmin = axobj.get_array().min()
                dmax = axobj.get_array().max()

        assert axobj is not None, "Failed to handle the provided data."
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(cbpos, size=cbsize, pad=cbpad)
            cbar = fig.colorbar(axobj, cax=cax)
            cbar.ax.set_yticks([dmin, dmax])
            cbar.ax.set_yticklabels(["{:4.2f}".format(i) for i in [dmin, dmax]])

        decorate_mpl_ax(
            fig=fig,
            ax=ax,
            points=points,
            title=title,
            suptitle=suptitle,
            label=label,
            **kwargs,
        )
        return axobj


__all__ = ["triplot_mpl_hinton", "triplot_mpl_mesh", "triplot_mpl_data"]
