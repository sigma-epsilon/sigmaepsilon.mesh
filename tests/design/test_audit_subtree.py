"""
Section 2.9 of the data model proposal: methods on a subtree don't line up.

These tests pin today's behaviour on a block holding two H8 cells and seven L2
cells. P5 (ids and methods on a subtree) changes it.
"""

import numpy as np
import pytest

from sigmaepsilon.mesh import PolyData
from sigmaepsilon.mesh.topoarray import TopologyArray

pytestmark = pytest.mark.audit("2.9")


def test_areas_drops_the_3d_blocks(mixed_mesh: PolyData):
    assert len(mixed_mesh.cell_ids()) == 9
    assert len(mixed_mesh.volumes()) == 9
    assert len(mixed_mesh.areas()) == 7


def test_ragged_topology_breaks_cells_coords(mixed_mesh: PolyData):
    from numba.core.errors import TypingError

    topology = mixed_mesh.topology()

    assert isinstance(topology, TopologyArray)
    assert sorted(set(topology.widths())) == [2, 8]
    # TypingError with Numba, IndexError with NUMBA_DISABLE_JIT=1
    with pytest.raises((TypingError, IndexError)):
        mixed_mesh.cells_coords()


def test_volume_counts_beams_with_unit_cross_section(mixed_mesh: PolyData):
    assert mixed_mesh["Solids"].volume() == pytest.approx(2.0)
    assert mixed_mesh["Beams"].volume() == pytest.approx(7.0)
    assert mixed_mesh.volume() == pytest.approx(9.0)


def test_cell_ids_of_a_subtree_come_from_a_method(mixed_mesh: PolyData):
    assert not hasattr(mixed_mesh, "id")
    assert np.all(mixed_mesh.cell_ids() == np.arange(9))
    assert np.all(mixed_mesh["Beams"].cd.id == np.arange(2, 9))
