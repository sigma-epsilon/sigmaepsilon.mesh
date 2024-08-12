from sigmaepsilon.core.downloads import download_file, delete_downloads
from sigmaepsilon.mesh import PolyData


__all__ = [
    "download_stand",
    "download_bunny",
    "delete_downloads",
    "download_bunny_coarse",
    "download_gt40",
    "download_badacsony",
    "download_bike_stem",
]


def _download(path: str, read: bool = False) -> str | PolyData:
    vtkpath = download_file(path)
    if read:
        return PolyData.read(vtkpath)
    else:
        return vtkpath


def download_stand(*, read: bool = False) -> str | PolyData:
    """
    Downloads a tetrahedral mesh of a stand in vtk format.

    Parameters
    ----------
    read: bool, Optional
        If False, the path of the mesh file is returned instead of a
        :class:`~sigmaepsilon.mesh.data.polydata.PolyData` object. Default is False.

    Returns
    -------
    str
        A path to a file on your filesystem.

    Example
    --------
    >>> from sigmaepsilon.mesh.downloads import download_stand
    >>> file_path = download_stand()
    """
    return _download("stand.vtk", read=read)


def download_bunny(*, tetra: bool = False, read: bool = False) -> str | PolyData:
    """
    Downloads a tetrahedral mesh of a bunny in vtk format.

    Parameters
    ----------
    tetra: bool, Optional
        If True, the returned mesh is a tetrahedral one, otherwise
        it is a surface triangulation. Default is False.
    read: bool, Optional
        If False, the path of the mesh file is returned instead of a
        :class:`~sigmaepsilon.mesh.data.polydata.PolyData` object. Default is False.

    Returns
    -------
    str
        A path to a file on your filesystem.

    Example
    --------
    >>> from sigmaepsilon.mesh.downloads import download_bunny
    >>> file_path = download_bunny()
    """
    filename = "bunny_T3.vtk" if not tetra else "bunny_TET4.vtk"
    return _download(filename, read=read)


def download_bunny_coarse(tetra: bool = False, read: bool = False) -> str | PolyData:
    """
    Downloads and optionally reads the bunny example as a vtk file.

    Parameters
    ----------
    tetra: bool, Optional
        If True, the returned mesh is a tetrahedral one, otherwise
        it is a surface triangulation. Default is False.
    read: bool, Optional
        If False, the path of the mesh file is returned instead of a
        :class:`~sigmaepsilon.mesh.data.polydata.PolyData` object. Default is False.

    Example
    -------
    >>> from sigmaepsilon.mesh.downloads import download_bunny_coarse
    >>> mesh = download_bunny_coarse(tetra=True, read=True)
    """
    filename = "bunny_T3_coarse.vtk" if not tetra else "bunny_TET4_coarse.vtk"
    return _download(filename, read=read)


def download_gt40(read: bool = False) -> str | PolyData:
    """
    Downloads and optionally reads the Gt40 example as a vtk file.

    Parameters
    ----------
    read: bool, Optional
        If False, the path of the mesh file is returned instead of a
        :class:`~sigmaepsilon.mesh.data.polydata.PolyData` object. Default is False.

    Example
    -------
    >>> from sigmaepsilon.mesh.downloads import download_gt40
    >>> mesh = download_gt40(read=True)
    """
    return _download("gt40.vtk", read=read)


def download_badacsony(read: bool = False) -> str | PolyData:
    """
    Downloads and optionally reads the badacsony example as a vtk file.

    Parameters
    ----------
    read: bool, Optional
        If False, the path of the mesh file is returned instead of a
        :class:`~sigmaepsilon.mesh.data.polydata.PolyData` object. Default is False.

    Example
    -------
    >>> from sigmaepsilon.mesh.downloads import download_badacsony
    >>> mesh = download_badacsony(read=True)
    """
    return _download("badacsony.vtk", read=read)


def download_bike_stem(read: bool = False) -> str | PolyData:
    """
    Downloads and optionally reads the bike stem example as an STL file.

    Parameters
    ----------
    read: bool, Optional
        If False, the path of the mesh file is returned instead of a
        :class:`~sigmaepsilon.mesh.data.polydata.PolyData` object. Default is False.

    Example
    -------
    >>> from sigmaepsilon.mesh.downloads import download_bike_stem
    >>> mesh = download_bike_stem(read=True)
    """
    return _download("bike_stem_nomanifold.stl", read=read)
