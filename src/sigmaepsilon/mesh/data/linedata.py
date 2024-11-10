from .polydata import PolyData
from ..config import __hasplotly__

if __hasplotly__:
    from ..plotting.plotly import plot_lines_plotly


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
        topo = self.topology().to_numpy()
        kwargs.update(dict(scalars=scalars, fig=fig))
        return plot_lines_plotly(coords, topo, **kwargs)

    def plot(self, *args, backend: str = "plotly", **kwargs):
        """
        Plots the mesh with or without data, using multiple possible backends.

        Parameters
        ----------
        backend: str, Optional
            The backend to use. Possible options are `plotly` and `vtk`
            (`vtk` is being managed through `PyVista`) at the moment.
            Default is 'plotly'.

        """
        if backend.lower() in ["vtk", "pyvista"]:
            return self.pvplot(*args, **kwargs)
        elif backend.lower() == "plotly":
            if __hasplotly__:
                return self.__plot_plotly__(*args, **kwargs)
            else:
                msg = "You need to install `plotly` for this."
                raise ImportError(msg)
        else:
            msg = "No implementation for backend '{}'".format(backend)
            raise NotImplementedError(msg)
