# -*- coding: utf-8 -*-
from typing import Iterable, Callable

import numpy as np
import matplotlib.pyplot as plt


def get_fig_axes(
    *args,
    data=None,
    fig=None,
    axes=None,
    shape=None,
    horizontal=False,
    ax=None,
    fig_kw=None,
    **kwargs,
) -> tuple:
    """
    Returns a figure and an axes object.
    """
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
                    aspect = kwargs.get("aspect", "equal")
                    args[0].set_aspect(aspect)
                    ax = args[0]
                except Exception:
                    fig, ax = plt.subplots(**fig_kw)
                return fig, (ax,)
            
            if fig is None or axes is None:
                if shape is not None:
                    if isinstance(shape, int):
                        shape = (shape, 1) if horizontal else (1, shape)
                    assert nD == (shape[0] * shape[1]), "Mismatch in shape and data."
                else:
                    shape = (nD, 1) if horizontal else (1, nD)
                fig, axes = plt.subplots(*shape, **fig_kw)
            
            if not isinstance(axes, Iterable):
                axes = (axes,)
            
            return fig, axes
        else:
            try:
                aspect = kwargs.get("aspect", "equal")
                args[0].set_aspect(aspect)
                ax = args[0]
            except Exception:
                fig, ax = plt.subplots(**fig_kw)
            return fig, (ax,)
    return None, None


def decorate_ax(
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
    **__,
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
