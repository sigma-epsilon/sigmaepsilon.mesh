from ...config import __hasplotly__

if __hasplotly__:
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

else:  # pragma: no cover

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


__all__ = ["plot_lines_plotly", "scatter_lines_plotly"]
