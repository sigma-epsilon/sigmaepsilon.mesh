# -*- coding: utf-8 -*-
from ...config import __hasmatplotlib__

if not __hasmatplotlib__:  # pragma: no cover

    def triplotter(*_, **__):
        raise ImportError(
            "You need Matplotlib for this. Install it with 'pip install matplotlib'. "
            "You may also need to restart your kernel and reload the package."
        )

    def get_fig_axes(*_, **__):
        raise ImportError(
            "You need Matplotlib for this. Install it with 'pip install matplotlib'. "
            "You may also need to restart your kernel and reload the package."
        )

    def decorate_mpl_ax(*_, **__):
        raise ImportError(
            "You need Matplotlib for this. Install it with 'pip install matplotlib'. "
            "You may also need to restart your kernel and reload the package."
        )

else:
    from typing import Iterable, Callable, Any
    from functools import wraps

    import numpy as np
    from numpy import ndarray
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure, Axes
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    from sigmaepsilon.mesh.triang import triangulate
    from sigmaepsilon.core.typing import issequence

    class TriPatchCollection(PatchCollection):
        def __init__(self, cellcoords, *args, **kwargs):
            pmap = map(lambda i: cellcoords[i], np.arange(len(cellcoords)))

            def fnc(points):
                return Polygon(points, closed=True)

            patches = list(map(fnc, pmap))
            super().__init__(patches, *args, **kwargs)

    def triplotter(plotter: Callable) -> Callable:
        @wraps(plotter)
        def inner(
            triobj: Any,
            *args,
            data: ndarray = None,
            title: str = None,
            label: str = None,
            fig: Figure = None,
            ax: Axes = None,
            axes: Iterable[Axes] = None,
            fig_kw: dict = None,
            **kwargs,
        ) -> Any:
            fig, axes = get_fig_axes(
                *args, data=data, ax=ax, axes=axes, fig=fig, fig_kw=fig_kw
            )

            if isinstance(triobj, tuple):
                coords, topo = triobj
                triobj = triangulate(points=coords[:, :2], triangles=topo)[-1]
                coords, topo = None, None

            if data is not None:
                if not len(data.shape) <= 2:
                    raise ValueError("Data must be a 1 or 2 dimensional array.")

                nD = 1 if len(data.shape) == 1 else data.shape[1]

                data = data.reshape((data.shape[0], nD))

                if not issequence(title):
                    title = nD * (title,)

                if not issequence(label):
                    label = nD * (label,)

                axobj = [
                    plotter(
                        triobj,
                        data[:, i],
                        fig=fig,
                        ax=ax,
                        title=title[i],
                        label=label[i],
                        **kwargs,
                    )
                    for i, ax in enumerate(axes)
                ]
                if nD == 1:
                    data = data.reshape(data.shape[0])
            else:
                axobj = plotter(triobj, ax=axes[0], fig=fig, title=title, **kwargs)

            return axobj

        return inner

    def get_fig_axes(
        *args,
        data=None,
        fig=None,
        axes=None,
        shape=None,
        horizontal=False,
        ax=None,
        fig_kw=None,
    ) -> tuple:
        """
        Returns a figure and an axes object.
        """
        if isinstance(ax, (tuple, list)):
            axes = ax

        if fig is not None:
            if axes is not None:
                return fig, axes
            elif ax is not None:
                return fig, (ax,)
        else:
            if fig_kw is None:
                fig_kw = {}

            if data is not None:
                nD = 1 if len(data.shape) == 1 else data.shape[1]

                if nD == 1:
                    try:
                        ax = args[0]
                    except Exception:
                        fig, ax = plt.subplots(**fig_kw)
                    return fig, (ax,)

                if fig is None or axes is None:
                    if shape is not None:
                        if isinstance(shape, int):
                            shape = (shape, 1) if horizontal else (1, shape)
                        assert nD == (
                            shape[0] * shape[1]
                        ), "Mismatch in shape and data."
                    else:
                        shape = (nD, 1) if horizontal else (1, nD)

                    fig, axes = plt.subplots(*shape, **fig_kw)

                if not isinstance(axes, Iterable):
                    axes = (axes,)

                return fig, axes
            else:
                try:
                    ax = args[0]
                except Exception:
                    fig, ax = plt.subplots(**fig_kw)
                return fig, (ax,)

        return None, None

    def decorate_mpl_ax(
        *,
        fig=None,
        ax=None,
        aspect="equal",
        xlim=None,
        ylim=None,
        axis="on",
        offset=0.05,
        points=None,
        axfnc: Callable = None,
        title=None,
        suptitle=None,
        label=None,
    ):
        """
        Decorates an axis using the most often used modifiers.
        """
        assert ax is not None, (
            "A matplotlib Axes object must be provided with " "keyword argument 'ax'!"
        )

        if axfnc is not None:
            try:
                axfnc(ax)
            except Exception:
                raise RuntimeError("Something went wrong when calling axfnc.")

        if xlim is None:
            if points is not None:
                xlim = points[:, 0].min(), points[:, 0].max()
                if offset is not None:
                    dx = np.abs(xlim[1] - xlim[0])
                    xlim = xlim[0] - offset * dx, xlim[1] + offset * dx

        if ylim is None:
            if points is not None:
                ylim = points[:, 1].min(), points[:, 1].max()
                if offset is not None:
                    dx = np.abs(ylim[1] - ylim[0])
                    ylim = ylim[0] - offset * dx, ylim[1] + offset * dx

        ax.set_aspect(aspect)
        ax.axis(axis)

        if xlim is not None:
            ax.set_xlim(*xlim)

        if ylim is not None:
            ax.set_ylim(*ylim)

        if title is not None:
            ax.set_title(title)

        if label is not None:
            ax.set_xlabel(label)

        if fig is not None and suptitle is not None:
            fig.suptitle(suptitle)

        return ax
