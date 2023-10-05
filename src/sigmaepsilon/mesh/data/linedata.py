import numpy as np
from contextlib import suppress

from .polydata import PolyData
from ..cells import L2, L3

from ..config import __hasplotly__

if __hasplotly__:
    with suppress(ImportError):
        from sigmaepsilon.plotting.plotly import plot_lines_3d


__all__ = ["LineData"]


class LineData(PolyData):
    """Data class for 1d cells."""

    def _init_config_(self):
        super()._init_config_()
        key = self.__class__._pv_config_key_
        self.config[key]["color"] = "k"
        self.config[key]["line_width"] = 10
        self.config[key]["render_lines_as_tubes"] = True

    def __plot_plotly__(self, *, scalars=None, fig=None, **kwargs):
        """
        Plots collections of lines and data provided on the nodes using `plotly`.

        Returns the figure object.
        """
        coords = self.coords()
        topo = self.topology()
        kwargs.update(dict(scalars=scalars, fig=fig))
        return plot_lines_3d(coords, topo, **kwargs)

    def plot(self, *args, scalars=None, backend="plotly", scalar_labels=None, **kwargs):
        """
        Plots the mesh with or without data, using multiple possible backends.

        Parameters
        ----------
        scalars: numpy.ndarray, Optional
            Stacked nodal information as an 1d or 2d NumPy array. Default is None.
        backend: str, Optional
            The backend to use. Possible options are `plotly` and `vtk`
            (`vtk` is being managed through `PyVista`) at the moment.
            Default is 'plotly'.
        scalar_labels: Iterable, Optional
            Labels for the datasets provided with 'scalars'. Default is None.
        """
        if backend == "vtk":
            return self.pvplot(
                *args, scalars=scalars, scalar_labels=scalar_labels, **kwargs
            )
        elif backend == "plotly":
            if __hasplotly__:
                return self.__plot_plotly__(
                    *args, scalars=scalars, scalar_labels=None, **kwargs
                )
            else:
                msg = "You need to install `dewloosh.plotly` for this plot."
                raise ImportError(msg)
        else:
            msg = "No implementation for backend '{}'".format(backend)
            raise NotImplementedError(msg)
