"""
Section 2.6 of the data model proposal: ids depend on where a block sits in the
tree. They come from the index managers of the root, are rewritten when a block
joins a tree and deleted when it leaves.

These tests pin today's behaviour. P5.1 (ids from per-block offsets) changes it.
"""

import numpy as np
import pytest

from sigmaepsilon.mesh import PolyData, PointData, CartesianFrame
from sigmaepsilon.mesh.cells import H8

pytestmark = pytest.mark.audit("2.6")


def test_ids_come_from_the_index_managers_of_the_root(mixed_mesh: PolyData):
    assert np.all(mixed_mesh["Solids"].cd.id == [0, 1])
    assert np.all(mixed_mesh["Beams"].cd.id == np.arange(2, 9))
    assert np.all(mixed_mesh.pd.id == np.arange(12))
    assert mixed_mesh.cim.next == 9
    assert mixed_mesh["Solids"].cim is None


def test_leaving_the_tree_deletes_the_stored_ids(mixed_mesh: PolyData):
    beams = mixed_mesh["Beams"]

    del mixed_mesh["Beams"]

    assert "_id" not in beams.cd.db.fields
    assert sorted(mixed_mesh.cim.queue) == list(range(2, 9))


def test_joining_a_tree_rewrites_the_stored_ids(mixed_mesh: PolyData):
    beams = mixed_mesh["Beams"]
    del mixed_mesh["Beams"]
    frame = CartesianFrame(dim=3)
    other = PolyData(PointData(coords=np.zeros((24, 3)), frame=frame))
    other["Solids"] = PolyData(H8(topo=np.arange(24).reshape(3, 8), frames=frame))

    other["Beams"] = beams

    assert np.all(beams.cd.id == np.arange(3, 10))
