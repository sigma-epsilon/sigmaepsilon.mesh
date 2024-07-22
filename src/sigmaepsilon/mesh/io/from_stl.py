from ..data import PolyData
from ..config import __haspyvista__, __haspymeshfix__
from ..helpers import importers
from .from_pyvista import from_pv

if __haspymeshfix__:
    from pymeshfix import MeshFix

if __haspyvista__:
    import pyvista as pv


if not __haspyvista__:  # pragma: no cover

    def from_stl(*_) -> PolyData:
        raise ImportError(
            "You need PyVista for this. Install it with 'pip install pyvista'. "
            "You may also need to restart your kernel and reload the package."
        )

else:

    def from_stl(
        stl_file_path: str,
        clean: bool = True,
        repair: bool = False,
        verbose: bool = False,
    ) -> PolyData:
        """
        Returns a :class:`~sigmaepsilon.mesh.polydata.PolyData` instance from
        an stl file.

        Parameters
        ----------
        stl_file_path : str
            The path to the STL file.
        clean : bool, optional
            Whether to clean the mesh using `PyVista`, by default True.
        repair : bool, optional
            Whether to fix the mesh using `PyMeshfix`, by default False.
        verbose : bool, optional
            Whether to print verbose output, by default False. Currently this
            only affects the PyMeshFix repair function.
        """
        # Read the STL file using PyVista
        mesh = pv.read(stl_file_path)

        if clean:
            # Ensure the mesh is manifold by using PyVista's cleaning functions
            mesh = mesh.clean()

        if repair:
            if not __haspymeshfix__:
                raise ImportError(
                    "You need PyMeshFix for this. Install it with 'pip install pymeshfix'. "
                    "You may also need to restart your kernel and reload the package."
                )

            # Further repair the mesh using PyMeshFix
            # Convert to a format suitable for PyMeshFix (vertices and faces)
            points = mesh.points
            # remove the first element of each face which is the number of points in the face
            faces = mesh.faces.reshape(-1, 4)[:, 1:4]

            # Initialize MeshFix
            meshfix = MeshFix(points, faces)

            # Perform the repair
            meshfix.repair(verbose=verbose)

            # Extract the repaired mesh
            mesh = meshfix.mesh

        # Convert the repaired mesh back to a PyVista PolyData object
        mesh = pv.PolyData(mesh.points, mesh.faces)

        if clean:
            mesh = mesh.clean()

        # Ensure the repaired mesh is cleaned and triangulated
        mesh = mesh.triangulate()

        return from_pv(mesh)


importers["stl"] = from_stl

__all__ = ["from_stl"]
