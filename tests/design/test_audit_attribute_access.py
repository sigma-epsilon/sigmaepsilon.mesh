"""
Section 2.3 of the data model proposal: attribute access falls through to
Awkward, and assigning an undeclared attribute doesn't store data.

These tests pin today's behaviour. P3.4 makes both warn, and P8.1 makes both
raise ``AttributeError``.
"""

import awkward as ak
import numpy as np
import pytest

from sigmaepsilon.mesh import PolyData

pytestmark = pytest.mark.audit("2.3")


def test_cell_block_falls_through_to_awkward(mixed_mesh: PolyData):
    cells = mixed_mesh["Solids"].cd

    # PolyCell -> CellData -> awkward.Array
    assert isinstance(cells.layout, ak.contents.Content)
    assert cells.ndim == 1


def test_undeclared_attribute_is_not_stored_as_data(mixed_mesh: PolyData):
    pd = mixed_mesh.pd

    pd.temperature = np.ones(len(pd))

    assert "temperature" in pd.__dict__
    assert "temperature" not in pd.db.fields
    assert "temperature" not in pd.to_dataframe().columns.get_level_values(0)
