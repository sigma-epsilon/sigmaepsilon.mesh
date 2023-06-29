from sigmaepsilon.core.downloads import (
    download_stand as _download_stand,
    download_bunny as _download_bunny,
    delete_downloads as _delete_downloads,
    _download_file,
)
from sigmaepsilon.mesh import PolyData
from typing import Union


__all__ = [
    "delete_downloads",
    "download_stand",
    "download_bunny",
    "download_bunny_coarse",
    "download_gt40",
]


def delete_downloads():
    """
    Delete all downloaded examples to free space or update the files.

    Returns
    -------
    bool
        Returns ``True`` if the operation was succesful, ``False`` otherwise.

    Examples
    --------
    Delete all local downloads.

    >>> from sigmaepsilon.mesh.examples import delete_downloads
    >>> delete_downloads()  # doctest:+SKIP
    True
    """
    return _delete_downloads()


def download_stand(read: bool = False) -> Union[str, PolyData]:
    """
    Downloads and optionally reads the stand example as a vtk file.

    Parameters
    ----------
    read: bool, Optional
        If False, the path of the mesh file is returned instead of a
        :class:`~sigmaepsilon.mesh.polydata.PolyData` object. Default is False.

    Example
    -------
    >>> from sigmaepsilon.mesh.examples import download_stand
    >>> mesh = download_stand(read=True)
    """
    vtkpath = _download_stand()
    if read:
        return PolyData.read(vtkpath)
    else:
        return vtkpath


def download_bunny(tetra: bool = False, read: bool = False) -> Union[str, PolyData]:
    """
    Downloads and optionally reads the bunny example as a vtk file.

    Parameters
    ----------
    tetra: bool, Optional
        If True, the returned mesh is a tetrahedral one, otherwise
        it is a surface triangulation. Default is False.
    read: bool, Optional
        If False, the path of the mesh file is returned instead of a
        :class:`~sigmaepsilon.mesh.polydata.PolyData` object. Default is False.

    Example
    -------
    >>> from sigmaepsilon.mesh.examples import download_bunny
    >>> mesh = download_bunny(tetra=True, read=True)
    """
    vtkpath = _download_bunny(tetra=tetra)
    if read:
        return PolyData.read(vtkpath)
    else:
        return vtkpath


def download_bunny_coarse(
    tetra: bool = False, read: bool = False
) -> Union[str, PolyData]:
    """
    Downloads and optionally reads the bunny example as a vtk file.

    Parameters
    ----------
    tetra: bool, Optional
        If True, the returned mesh is a tetrahedral one, otherwise
        it is a surface triangulation. Default is False.
    read: bool, Optional
        If False, the path of the mesh file is returned instead of a
        :class:`~sigmaepsilon.mesh.polydata.PolyData` object. Default is False.

    Example
    -------
    >>> from sigmaepsilon.mesh.examples import download_bunny_coarse
    >>> mesh = download_bunny_coarse(tetra=True, read=True)
    """
    filename = "bunny_T3_coarse.vtk" if not tetra else "bunny_TET4_coarse.vtk"
    vtkpath = _download_file(filename)[0]
    if read:
        return PolyData.read(vtkpath)
    else:
        return vtkpath


def download_gt40(read: bool = False) -> Union[str, PolyData]:
    """
    Downloads and optionally reads the Gt40 example as a vtk file.

    Parameters
    ----------
    read: bool, Optional
        If False, the path of the mesh file is returned instead of a
        :class:`~sigmaepsilon.mesh.polydata.PolyData` object. Default is False.

    Example
    -------
    >>> from sigmaepsilon.mesh.examples import download_gt40
    >>> mesh = download_gt40(read=True)
    """
    filename = "gt40.vtk"
    vtkpath = _download_file(filename)[0]
    if read:
        return PolyData.read(vtkpath)
    else:
        return vtkpath


def download_badacsony(read: bool = False) -> Union[str, PolyData]:
    """
    Downloads and optionally reads the badacsony example as a vtk file.

    Parameters
    ----------
    read: bool, Optional
        If False, the path of the mesh file is returned instead of a
        :class:`~sigmaepsilon.mesh.polydata.PolyData` object. Default is False.

    Example
    -------
    >>> from sigmaepsilon.mesh.examples import download_badacsony
    >>> mesh = download_badacsony(read=True)
    """
    filename = "badacsony.vtk"
    vtkpath = _download_file(filename)[0]
    if read:
        return PolyData.read(vtkpath)
    else:
        return vtkpath
