"""
Section 2.2 of the data model proposal: subclasses can't reliably extend the
attribute map, because ``ABCMeta_AkWrapper`` merges ``_attr_map_`` so that the
base class wins.

This is a bug: the test asserts the target behaviour and is a strict ``xfail``
until P2.3 merges the maps with the subclass winning.
"""

import pytest

from sigmaepsilon.mesh.data.celldata import CellData

pytestmark = pytest.mark.audit("2.2")


@pytest.mark.xfail(strict=True, reason="base class wins the merge, fixed in P2.3")
def test_subclass_entry_of_attribute_map_wins():
    # what solid.fem's CellData declares
    class FemLikeCellData(CellData):
        _attr_map_ = {"activity": "_activity_"}

    assert FemLikeCellData._attr_map_["activity"] == "_activity_"
