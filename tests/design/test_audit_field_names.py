"""
Section 2.1 of the data model proposal: internal and user field names can clash.

These tests pin today's behaviour. P3 (core fields and user fields) changes it.
"""

import numpy as np
import pytest

from sigmaepsilon.mesh import PolyData
from sigmaepsilon.mesh.data.celldata import CellData

pytestmark = pytest.mark.audit("2.1")


def test_point_and_cell_data_use_different_prefixes(mixed_mesh: PolyData):
    assert mixed_mesh.pd.db.fields == ["__x", "__id"]
    assert mixed_mesh["Solids"].cd.db.fields == ["_topology", "_frames", "_id"]


def test_writing_an_internal_name_overwrites_the_global_ids(mixed_mesh: PolyData):
    db = mixed_mesh["Solids"].cd.db
    assert np.all(db.id == [0, 1])

    db["_id"] = np.zeros(2, dtype=int)

    assert np.all(db.id == [0, 0])


def test_read_by_key_is_mapped_but_write_by_key_is_not(mixed_mesh: PolyData):
    db = mixed_mesh["Solids"].cd.db
    topology = db.nodes
    users_array = np.full((2, 8), 99)

    db["nodes"] = users_array

    assert "nodes" in db.fields
    assert np.all(db["nodes"].to_numpy() == topology)


def test_fields_argument_remaps_public_names_onto_internal_fields():
    topology = np.array([[0, 1, 2], [1, 2, 3]])
    cd = CellData(
        topo=topology,
        fields={"t": np.array([5.0, 6.0]), "id": np.array([7, 8])},
    )

    assert "t" not in cd.fields and "id" not in cd.fields
    assert np.all(cd.t == [5.0, 6.0])
    assert np.all(cd.id == [7, 8])


def test_internal_names_leak_into_fields_and_exports(mixed_mesh: PolyData, tmp_path):
    import awkward as ak

    pd = mixed_mesh.pd
    cd = mixed_mesh["Solids"].cd.db

    assert "__x" in pd.fields
    assert "_topology" in cd.fields
    assert "__x" in pd.to_dataframe().columns.get_level_values(0)
    assert "_topology" in cd.to_dataframe().columns.get_level_values(0)

    path = tmp_path / "cells.parquet"
    cd.to_parquet(str(path))
    assert "_topology" in ak.from_parquet(str(path)).fields
