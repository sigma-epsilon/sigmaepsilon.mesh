from ...config import __hasplotly__

if not __hasplotly__:  # pragma: no cover

    def scatter_points_plotly(*_, **__):
        raise ImportError(
            "You need Plotly for this. Install it with 'pip install plotly'. "
            "You may also need to restart your kernel and reload the package."
        )

else:
    from typing import Iterable, Optional, Union
    from numbers import Number

    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np
    from numpy import ndarray

    from sigmaepsilon.math import atleast2d

    def scatter_points_plotly(
        coords: Optional[ndarray],
        *,
        scalars: Optional[Union[ndarray, None]] = None,
        markersize: Optional[Number] = 1,
        scalar_labels: Optional[Union[Iterable[str], None]] = None,
    ) -> go.Figure:
        """
        Convenience function to plot several points in 3d with data and labels.

        Parameters
        ----------
        coords: numpy.ndarray
            The coordinates of the points, where the first axis runs along the points, the
            second along spatial dimensions.
        scalars: numpy.ndarray
            The values to show in the tooltips of the points as a 1d or 2d NumPy array.
            The length of the array must equal the number of points. Default is None.
        markersize: int, Optional
            The size of the balls at the point coordinates. Default is 1.
        scalar_labels: Iterable[str], Optional
            The labels for the columns in 'scalars'. Default is None.

        Example
        -------
        .. plotly::
            :include-source: True

            from sigmaepsilon.mesh.plotting import scatter_points_plotly
            import numpy as np
            points = np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
            data = np.random.rand(len(points))
            scalar_labels=["random data"]
            scatter_points_plotly(points, scalars=data, scalar_labels=scalar_labels)

        """
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        dfdata = {
            "x": x,
            "y": y,
            "z": z,
            "size": np.full(len(x), markersize),
            "symbol": np.full(len(x), 4),
            "id": np.arange(1, len(x) + 1),
        }

        if scalars is not None:
            scalars = atleast2d(scalars, back=True)

        if scalars is not None and scalar_labels is None:
            nData = scalars.shape[1]
            scalar_labels = ["#{}".format(i + 1) for i in range(nData)]

        if scalar_labels is not None:
            assert len(scalar_labels) == scalars.shape[1]
            hover_data = scalar_labels
            sdata = {scalar_labels[i]: scalars[:, i] for i in range(len(scalar_labels))}
            dfdata.update(sdata)
        else:
            hover_data = ["x", "y", "z"]

        custom_data = scalar_labels if scalars is not None else None

        df = pd.DataFrame.from_dict(dfdata)

        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            hover_name="id",
            hover_data=hover_data,
            size="size",
            text="id",
            custom_data=custom_data,
        )

        if scalars is not None:
            tmpl = lambda i: "{" + "customdata[{}]:.4e".format(i) + "}"
            lbl = lambda i: scalar_labels[i]
            fnc = lambda i: "{label}: %{index}".format(label=lbl(i), index=tmpl(i))
            labels = [fnc(i) for i in range(len(scalar_labels))]
            fig.update_traces(hovertemplate="<br>".join(labels))
        else:
            fig.update_traces(
                hovertemplate="<br>".join(
                    [
                        "x: %{x}",
                        "y: %{y}",
                        "z: %{z}",
                    ]
                )
            )

        return fig


__all__ = ["scatter_points_plotly"]
