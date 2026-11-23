"""
Fixtures for the characterization tests of the data model proposal.

The tests in this folder pin the findings of section 2 of
``design/data-model-proposal.md`` (work package P0.2 of
``design/data-model-implementation-plan.md``). A test that asserts today's
behaviour is edited by the work package that changes it. A test marked
``xfail(strict=True)`` asserts the target behaviour of a bug and loses its
marker in the work package that fixes the bug.
"""

import numpy as np
import pytest

from sigmaepsilon.mesh import PolyData, PointData, CartesianFrame
from sigmaepsilon.mesh.cells import H8, L2
from sigmaepsilon.mesh.grid import grid

# the 7 edges of the bottom face of the 2 x 1 x 1 box
BEAM_TOPOLOGY = np.array([[0, 4], [4, 8], [2, 6], [6, 10], [0, 2], [4, 6], [8, 10]])


@pytest.fixture
def mixed_mesh() -> PolyData:
    """
    Two H8 cells on a 2 x 1 x 1 box and seven L2 cells along its bottom
    edges, all sharing the 12 points held by the root (section 2.9).
    """
    frame = CartesianFrame(dim=3)
    coords, topology = grid(size=(2, 1, 1), shape=(2, 1, 1), eshape="H8")
    mesh = PolyData(PointData(coords=coords, frame=frame))
    mesh["Solids"] = PolyData(H8(topo=topology, frames=frame))
    mesh["Beams"] = PolyData(L2(topo=BEAM_TOPOLOGY, frames=frame))
    return mesh
