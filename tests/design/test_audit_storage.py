"""
Section 2.4 of the data model proposal: Awkward is mostly used as a dict of
regular columns.

These tests pin today's behaviour. P3 (per-field columns) and P4 (frame
storage) change it.
"""

import numpy as np
import pytest

from sigmaepsilon.mesh import PolyData, CartesianFrame
from sigmaepsilon.mesh.cells import L2

pytestmark = pytest.mark.audit("2.4")


def test_shared_frame_is_stored_expanded_per_cell():
    frame = CartesianFrame(dim=3)
    topology = np.stack([np.arange(30), np.arange(1, 31)], axis=1)

    cells = L2(topo=topology, frames=frame)

    assert cells.frames.shape == (30, 3, 3)
    assert cells.frames.nbytes == 2160


def test_blocks_with_different_fields_export_as_optional_types(
    mixed_mesh: PolyData,
):
    mixed_mesh["Solids"].cd.db["mark"] = np.ones(2)

    _, cells = mixed_mesh.to_ak()

    assert "mark: ?float64" in str(cells.type)
