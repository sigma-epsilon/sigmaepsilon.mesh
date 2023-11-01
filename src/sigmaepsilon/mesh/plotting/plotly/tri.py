from ...config import __hasplotly__

if not __hasplotly__:  # pragma: no cover

    def triplot_plotly(*_, **__):
        raise ImportError(
            "You need Plotly for this. Install it with 'pip install plotly'. "
            "You may also need to restart your kernel and reload the package."
        )

else:
    from typing import Optional, Union

    import plotly.graph_objects as go
    from numpy import ndarray

    from sigmaepsilon.mesh.utils.topology import unique_topo_data
    from sigmaepsilon.mesh.utils.tri import edges_tri
    from .lines import scatter_lines_plotly

    def triplot_plotly(
        points: ndarray,
        triangles: ndarray,
        data: Optional[ndarray] = None,
        plot_edges: Optional[bool] = True,
        edges: Optional[Union[bool, None]] = None,
    ) -> go.Figure:
        """
        Plots a triangulation optionally with edges and attached field data in 3d.

        Parameters
        ----------
        points: numpy.ndarray
            2d float array of points.
        data: numpy.ndarray
            1d float array of scalar data over the points.
        plot_edges: bool, Optional
            If True, plots the edges of the mesh. Default is False.
        edges: numpy.ndarray, Optional
            The edges to plot. If provided, `plot_edges` is ignored. Default is None.

        Returns
        -------
        figure: :class:`plotly.graph_objects.Figure`
            The figure object.

        Example
        -------
        .. plotly::
            :include-source: True

            from sigmaepsilon.mesh.plotting import triplot_plotly
            from sigmaepsilon.mesh import grid
            from sigmaepsilon.mesh.utils.topology.tr import Q4_to_T3
            import numpy as np
            gridparams = {
                "size": (1200, 600),
                "shape": (4, 4),
                "eshape": (2, 2),
            }
            coords, topo = grid(**gridparams)
            points, triangles = Q4_to_T3(coords, topo, path="grid")
            data = np.random.rand(len(points))
            triplot_plotly(points, triangles, data, plot_edges=True)
        """
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        i = triangles[:, 0]
        j = triangles[:, 1]
        k = triangles[:, 2]

        if data is not None:
            fig = go.Figure(
                data=[
                    go.Mesh3d(
                        x=x,
                        y=y,
                        z=z,
                        i=i,
                        j=j,
                        k=k,
                        intensity=data,
                        opacity=1,
                    )
                ]
            )
        else:
            fig = go.Figure(
                data=[
                    go.Mesh3d(
                        x=x,
                        y=y,
                        z=z,
                        i=i,
                        j=j,
                        k=k,
                        opacity=1,
                    )
                ]
            )

        fig.update_layout(
            template="plotly",
            autosize=True,
            margin=dict(l=1, r=1, b=1, t=1, pad=0),
            scene=dict(
                aspectmode="data",
            ),
        )

        if edges is not None:
            plot_edges = True

        if plot_edges:
            if edges is None:
                edges, _ = unique_topo_data(edges_tri(triangles))

            scatter_lines_plotly(points, edges, fig=fig)

        return fig


__all__ = ["triplot_plotly"]
