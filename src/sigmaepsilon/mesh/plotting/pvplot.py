from ..config import __haspyvista__
from ..helpers import plotters

if not __haspyvista__:  # pragma: no cover

    def pvplot(*_, **__) -> None:
        raise ImportError(
            "You need PyVista for this. Install it with 'pip install pyvista'. "
            "You may also need to restart your kernel and reload the package."
        )

else:
    from typing import Union, Iterable, Tuple
    from types import NoneType

    import pyvista as pv
    from pyvista import themes
    from numpy import ndarray

    from ..data import PolyData

    def pvplot(
        obj: PolyData,
        *,
        deepcopy: bool = False,
        jupyter_backend: str = "pythreejs",
        show_edges: bool = True,
        notebook: bool = False,
        theme: str | NoneType = None,
        scalars: Union[str, ndarray, NoneType] = None,
        window_size: Tuple | NoneType = None,
        return_plotter: bool = False,
        config_key: Tuple | NoneType = None,
        plotter: pv.Plotter | NoneType = None,
        cmap: Union[str, Iterable, NoneType] = None,
        camera_position: Tuple | NoneType = None,
        lighting: bool = False,
        edge_color: str | NoneType = None,
        return_img: bool = False,
        show_scalar_bar: Union[bool, NoneType] = None,
        opacity: float | NoneType = None,
        style: str | NoneType = None,
        add_legend: bool = False,
        legend_params: dict | NoneType = None,
        **kwargs,
    ) -> Union[None, pv.Plotter, ndarray]:
        """
        Plots the mesh using PyVista. The parameters listed here only grasp
        a fraction of what PyVista provides. The idea is to have a function
        that narrows down the parameters as much as possible to the ones that
        are most commonly used. If you want more control, create a plotter
        prior to calling this function and provide it using the parameter
        `plotter`. Then by setting `return_plotter` to `True`, the function
        adds the cells to the plotter and returns it for further customization.

        Parameters
        ----------
        deepcopy: bool, Optional
            If True, a deep copy is created. Default is False.
        jupyter_backend: str, Optional
            The backend to use when plotting in a Jupyter enviroment.
            Default is 'pythreejs'.
        show_edges: bool, Optional
            If True, the edges of the mesh are shown as a wireframe.
            Default is True.
        notebook: bool, Optional
            If True and in a Jupyter enviroment, the plot is embedded
            into the Notebook. Default is False.
        theme: str, Optional
            The theme to use with PyVista. Default is None.
        scalars: Union[str, numpy.ndarray]
            A string that refers to a field in the celldata objects
            of the block of the mesh, or a NumPy array with values for
            each point in the mesh.
        window_size: tuple, Optional
            The size of the window, only is `notebook` is `False`.
            Default is None.
        return_plotter: bool, Optional
            If True, an instance of :class:`pyvista.Plotter` is returned
            without being shown. Default is False.
        config_key: tuple, Optional
            A tuple of strings that refer to a configuration for PyVista.
        plotter: pyvista.Plotter, Optional
            A plotter to use. If not provided, a plotter is created in the
            background. Default is None.
        cmap: Union[str, Iterable], Optional
            A color map for plotting. See PyVista's docs for the details.
            Default is None.
        camera_position: tuple, Optional
            Camera position. See PyVista's docs for the details. Default is None.
        lighting: bool, Optional
            Whether to use lighting or not. Default is None.
        edge_color: str, Optional
            The color of the edges if `show_edges` is `True`. Default is None,
            which equals to the default PyVista setting.
        return_img: bool, Optional
            If True, a screenshot is returned as an image. Default is False.
        show_scalar_bar: Union[bool, None], Optional
            Whether to show the scalar bar or not. A `None` value means that the option
            is governed by the configurations of the blocks. If a boolean is provided here,
            it overrides the configurations of the blocks. Default is None.
        opacity: float, Optional
            The opacity of the mesh. Default is None.
            .. versionadded:: 3.1.0
        style: str, Optional
            Visualization style of the mesh. Default is None.
            .. versionadded:: 3.1.0
        add_legend: bool, Optional
            If True, a legend is added to the plot. Default is False.
            .. versionadded:: 3.1.0
        legend_params: dict, Optional
            Parameters for the legend to be forwarded to the function
            :func:`pyvista.Plotter.add_legend()`. Default is None.
            .. versionadded:: 3.1.0
        **kwargs
            Extra keyword arguments passed to `pyvista.Plotter`, if the plotter
            has to be created.

        Returns
        -------
        Union[None, pv.Plotter, numpy.ndarray]
            A PyVista plotter if `return_plotter` is `True`, a NumPy array if
            `return_img` is `True`, or nothing.

        Example
        -------
        .. plot::
            :include-source: True

            from sigmaepsilon.mesh.plotting import pvplot
            from sigmaepsilon.mesh.downloads import download_gt40
            import matplotlib.pyplot as plt
            mesh = download_gt40(read=True)
            img=pvplot(mesh, notebook=False, return_img=True)
            plt.imshow(img)
            plt.axis('off')

        """
        if not isinstance(obj, PolyData):  # pragma: no cover
            raise TypeError(f"Expected PolyData, got {type(obj)}.")

        polys = obj.to_pv(deepcopy=deepcopy, multiblock=False, scalars=scalars)

        if isinstance(theme, str):
            try:
                new_theme_type = pv.themes._ALLOWED_THEMES[theme].value
                theme = new_theme_type()
            except Exception:
                if theme == "dark":
                    theme = themes.DarkTheme()
                    theme.lighting = False
                elif theme == "bw":
                    theme = themes.Theme()
                    theme.color = "black"
                    theme.lighting = True
                    theme.edge_color = "white"
                    theme.background = "white"
                elif theme == "document":
                    theme = themes.DocumentTheme()
                else:
                    raise ValueError(f"Unknown theme: {theme}")

        if theme is None:
            theme = pv.global_theme

        if show_edges is not None:
            theme.show_edges = show_edges

        if lighting is not None:
            theme.lighting = lighting

        if edge_color is not None:
            theme.edge_color = edge_color

        if plotter is None:
            pvparams = dict()

            if window_size is not None:
                pvparams.update(window_size=window_size)

            pvparams.update(kwargs)
            pvparams.update(notebook=notebook)
            pvparams.update(theme=theme)

            if "title" not in pvparams:
                pvparams["title"] = "SigmaEpsilon"

            if not notebook and return_img:
                pvparams["off_screen"] = True

            plotter = pv.Plotter(**pvparams)

        if camera_position is not None:
            plotter.camera_position = camera_position

        blocks: Iterable[PolyData] = obj.cellblocks(inclusive=True, deep=True)

        blocks_have_data = obj._has_plot_scalars_(scalars)

        if config_key is None:
            config_key = obj.__class__._pv_config_key_

        for block, poly, has_data in zip(blocks, polys, blocks_have_data):
            num_dim = block.cd.Geometry.number_of_spatial_dimensions
            params = dict()
            config = block._get_config_(config_key)

            if has_data:
                config.pop("color", None)

            params.update(config)

            if cmap is not None:
                params["cmap"] = cmap

            if (
                (num_dim > 1)
                and (show_edges is not None)
                and ("show_edges" not in params)
            ):
                params["show_edges"] = show_edges

            if isinstance(opacity, float) and ("opacity" not in params):
                params["opacity"] = opacity

            if "style" not in params:
                params["style"] = style

            if isinstance(show_scalar_bar, bool):
                params["show_scalar_bar"] = show_scalar_bar

            plotter.add_mesh(poly, **params)

        if add_legend:
            if legend_params is None:
                legend_params = dict()
            plotter.add_legend(**legend_params)

        if return_plotter:
            return plotter

        show_params = dict()
        if notebook:
            show_params.update(jupyter_backend=jupyter_backend)
        else:
            if return_img:
                plotter.show(auto_close=False)
                plotter.show(screenshot=True)
                return plotter.last_image

        return plotter.show(**show_params)


plotters["PyVista"] = pvplot

__all__ = ["pvplot"]
