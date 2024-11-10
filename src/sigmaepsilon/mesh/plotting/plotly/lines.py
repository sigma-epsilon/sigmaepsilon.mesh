from ...config import __hasplotly__

if not __hasplotly__:  # pragma: no cover

    def plot_lines_plotly(*_, **__):
        raise ImportError(
            "You need Plotly for this. Install it with 'pip install plotly'. "
            "You may also need to restart your kernel and reload the package."
        )

    def scatter_lines_plotly(*_, **__):
        raise ImportError(
            "You need Plotly for this. Install it with 'pip install plotly'. "
            "You may also need to restart your kernel and reload the package."
        )

else:
    import plotly.graph_objects as go
    from numpy import ndarray

    from sigmaepsilon.mesh.utils import explode_mesh_bulk

    from .points import scatter_points_plotly

    def scatter_lines_plotly(
        coords: ndarray, topo: ndarray, fig: go.Figure = None
    ) -> go.Figure:
        X, _ = explode_mesh_bulk(coords, topo)
        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]

        if fig is None:
            fig = go.Figure()

        def _stack_line_3d(i):
            scatter_cells = go.Scatter3d(
                x=x[2 * i : 2 * (i + 1)],
                y=y[2 * i : 2 * (i + 1)],
                z=z[2 * i : 2 * (i + 1)],
                mode="lines",
                line=dict(color="black", width=4),
                showlegend=False,
            )
            scatter_cells.update(hoverinfo="skip")
            fig.add_trace(scatter_cells)

        list(map(_stack_line_3d, range(topo.shape[0])))

        return fig

    def plot_lines_plotly(
        coords: ndarray,
        topo: ndarray,
        *args,
        scalars: ndarray = None,
        fig: go.Figure = None,
        marker_symbol: str = "circle",
        **kwargs,
    ) -> go.Figure:
        """
        Plots points and lines in 3d space optionally with data defined on the points.
        If data is provided, the values are shown in a tooltip when howering above a point.

        .. note:
            Currently only 2 noded linear lines are supported.

        Parameters
        ----------
        coords: numpy.ndarray
            The coordinates of the points, where the first axis runs along the points, the
            second along spatial dimensions.
        topo: numpy.ndarray
            The topology of the lines, where the first axis runs along the lines, the
            second along the nodes.
        scalars: numpy.ndarray
            The values to show in the tooltips of the points as a 1d or 2d NumPy array.
            The length of the array must equal the number of points. Default is None.
        marker_symbol: str, Optional
            The symbol to use for the points. Refer to Plotly's documentation for the
            possible options. Default is "circle".

        Example
        -------
        .. plotly::
            :include-source: True

            from sigmaepsilon.mesh.plotting import plot_lines_plotly
            from sigmaepsilon.mesh import grid
            from sigmaepsilon.mesh.utils.topology.tr import H8_to_L2
            import numpy as np
            gridparams = {
                "size": (10, 10, 10),
                "shape": (4, 4, 4),
                "eshape": "H8",
            }
            coords, topo = grid(**gridparams)
            coords, topo = H8_to_L2(coords, topo)
            data = np.random.rand(len(coords), 2)
            plot_lines_plotly(coords, topo, scalars=data, scalar_labels=["X", "Y"])

        """
        n2 = topo[:, [0, -1]].max() + 1
        _scalars = scalars[:n2] if scalars is not None else None

        if fig is None:
            fig = scatter_points_plotly(coords[:n2], *args, scalars=_scalars, **kwargs)
        else:
            scatter_points_plotly(
                coords[:n2], *args, scalars=_scalars, fig=fig, **kwargs
            )

        for i, _ in enumerate(fig.data):
            fig.data[i].marker.symbol = marker_symbol
            fig.data[i].marker.size = 5

        scatter_lines_plotly(coords, topo, fig=fig)

        fig.update_layout(
            template="plotly",
            autosize=True,
            # width=720,
            # height=250,
            margin=dict(l=1, r=1, b=1, t=1, pad=0),
            scene=dict(
                aspectmode="data",
                # xaxis = dict(nticks=5, range=[xmin - delta, xmax + delta],),
                # yaxis = dict(nticks=5, range=[ymin - delta, ymax + delta],),
                # zaxis = dict(nticks=5, range=[zmin - delta, zmax + delta],),
            ),
        )

        return fig


__all__ = ["plot_lines_plotly", "scatter_lines_plotly"]
