import importlib.util


def is_module_available(module_name):
    return importlib.util.find_spec(module_name) is not None


__hasvtk__ = is_module_available("vtk")
__haspyvista__ = is_module_available("pyvista")
__hasmatplotlib__ = is_module_available("matplotlib")
__hasplotly__ = is_module_available("plotly")
__hasnx__ = is_module_available("networkx")
__hask3d__ = is_module_available("k3d")
