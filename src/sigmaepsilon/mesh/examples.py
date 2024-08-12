from sigmaepsilon.mesh.space import CartesianFrame
from sigmaepsilon.mesh.recipes import cylinder
from sigmaepsilon.mesh import PolyData, LineData, PointData
from sigmaepsilon.mesh.cells import H8, Q4, L2
from sigmaepsilon.mesh.utils.topology import H8_to_L2, H8_to_Q4
from sigmaepsilon.mesh.utils.topology import detach_mesh_bulk
from sigmaepsilon.mesh.utils.space import frames_of_lines, frames_of_surfaces
from sigmaepsilon.math import minmax
import numpy as np


def compound_mesh() -> PolyData:
    """
    Returns a compound mesh with H8, Q4 and L2 cells in it.
    """
    min_radius = 5
    max_radius = 25
    h = 50

    shape = (min_radius, max_radius), h
    frame = CartesianFrame(dim=3)
    cyl = cylinder(shape, size=5.0, voxelize=True, frame=frame)

    coords = cyl.coords()
    topo = cyl.topology().to_numpy()
    centers = cyl.centers()

    cxmin, cxmax = minmax(centers[:, 0])
    czmin, czmax = minmax(centers[:, 2])
    cxavg = (cxmin + cxmax) / 2
    czavg = (czmin + czmax) / 2
    b_upper = centers[:, 2] > czavg
    b_lower = centers[:, 2] <= czavg
    b_left = centers[:, 0] < cxavg
    b_right = centers[:, 0] >= cxavg
    iL2 = np.where(b_upper & b_right)[0]
    iTET4 = np.where(b_upper & b_left)[0]
    iH8 = np.where(b_lower)[0]
    _, tL2 = H8_to_L2(coords, topo[iL2])
    _, tQ4 = H8_to_Q4(coords, topo[iTET4])
    tH8 = topo[iH8]

    pd = PointData(coords=coords, frame=frame)
    mesh = PolyData(pd)

    cdL2 = L2(topo=tL2, frames=frames_of_lines(coords, tL2))
    mesh["lines", "L2"] = LineData(cdL2)

    cdQ4 = Q4(topo=tQ4, frames=frames_of_surfaces(coords, tQ4))
    mesh["surfaces", "Q4"] = PolyData(cdQ4)

    cH8, tH8 = detach_mesh_bulk(coords, tH8)
    pdH8 = PointData(coords=cH8, frame=frame)
    cdH8 = H8(topo=tH8, frames=frame)
    mesh["bodies", "H8"] = PolyData(pdH8, cdH8)

    mesh.to_standard_form()
    mesh.lock(create_mappers=True)

    return mesh
