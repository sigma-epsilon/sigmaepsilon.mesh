"""
Section 2.7 of the data model proposal: nothing round-trips through a file.

The pickle test asserts the target behaviour and are strict
``xfail`` until P1.5. The parquet test pins today's behaviour, which P7
(persistence) replaces.
"""

import pickle

import numpy as np
import pytest

from sigmaepsilon.mesh import PolyData

pytestmark = pytest.mark.audit("2.7")


@pytest.mark.xfail(strict=True, reason="DeepDict.__reduce__ drops pd and cd, P1.5")
def test_pickle_keeps_point_and_cell_data(mixed_mesh: PolyData):
    loaded = pickle.loads(pickle.dumps(mixed_mesh))

    assert loaded.pd is not None
    assert np.allclose(loaded.pd.x, mixed_mesh.pd.x)
    assert loaded["Solids"].cd is not None
    assert np.all(loaded["Solids"].cd.nodes == mixed_mesh["Solids"].cd.nodes)


def test_parquet_export_has_no_reader(mixed_mesh: PolyData, tmp_path):
    mixed_mesh.to_parquet(
        str(tmp_path / "points.parquet"), str(tmp_path / "cells.parquet")
    )

    assert (tmp_path / "points.parquet").exists()
    assert (tmp_path / "cells.parquet").exists()
    assert not hasattr(PolyData, "from_parquet")
