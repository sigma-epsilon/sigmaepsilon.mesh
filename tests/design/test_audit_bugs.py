"""
Section 2.8 of the data model proposal: bugs found along the way.

Every test asserts the target behaviour and is a strict ``xfail`` until the
work package of P1 named in its reason fixes the bug.
"""

import numpy as np
import pytest

from sigmaepsilon.mesh import PolyData
from sigmaepsilon.mesh.indexmanager import IndexManager
from sigmaepsilon.mesh.space import PointCloud

pytestmark = pytest.mark.audit("2.8")


@pytest.mark.xfail(strict=True, reason="raises KeyError: '_ndf', P1.1")
def test_set_nodal_distribution_factors(mixed_mesh: PolyData):
    cd = mixed_mesh["Solids"].cd.db
    factors = np.full((2, 8), 0.5)

    cd.set_nodal_distribution_factors(factors)

    assert np.allclose(cd.ndf.to_numpy(), factors)


@pytest.mark.xfail(strict=True, reason="looks up _ndf and indexes with .id, P1.1")
def test_set_nodal_distribution_factors_of_all_cells(mixed_mesh: PolyData):
    # factors for every cell of the mesh, indexed by global cell id
    cd = mixed_mesh["Beams"].cd.db
    factors = np.arange(9 * 2, dtype=float).reshape(9, 2)

    cd.set_nodal_distribution_factors(factors)

    assert np.allclose(cd.ndf.to_numpy(), factors[2:9])


@pytest.mark.xfail(strict=True, reason="extends the caller's list, P1.2")
@pytest.mark.parametrize("method", ["to_dict", "to_list"])
def test_export_does_not_modify_fields_argument(mixed_mesh: PolyData, method):
    fields = ["_topology"]

    getattr(mixed_mesh["Solids"].cd.db, method)("_id", fields=fields)

    assert fields == ["_topology"]


@pytest.mark.xfail(strict=True, reason="calls extend on a tuple, P1.2")
@pytest.mark.parametrize("method", ["to_dict", "to_list"])
def test_export_accepts_a_tuple_of_fields(mixed_mesh: PolyData, method):
    getattr(mixed_mesh["Solids"].cd.db, method)("_id", fields=("_topology",))


def test_index_manager_hands_out_ids_twice_today():
    """The numbers quoted in section 2.8, edited by P1.3."""
    im = IndexManager()
    im.generate(10)
    im.recycle(range(10))

    assert im.generate(4) == [0, 1, 2, 3]
    assert im.generate(10) == [4, 5, 6, 7, 8, 9, 4, 5, 6, 7]


@pytest.mark.xfail(strict=True, reason="moves the counter backwards, P1.3")
def test_index_manager_never_hands_out_an_id_twice():
    im = IndexManager()
    im.generate(10)
    im.recycle(range(10))

    ids = im.generate(4) + im.generate(10)

    assert len(set(ids)) == len(ids) == 14


@pytest.mark.xfail(strict=True, reason="recycle accepts any id, P1.3")
@pytest.mark.parametrize(
    "ids",
    [(5,), (-1,), ((1, 1),), (1, 1)],
    ids=["never-generated", "negative", "twice-in-a-list", "twice-as-args"],
)
def test_index_manager_rejects_ids_not_in_use(ids):
    im = IndexManager()
    im.generate(3)

    with pytest.raises(ValueError):
        im.recycle(*ids)


@pytest.fixture
def cloud() -> tuple[np.ndarray, PointCloud]:
    coords = np.random.default_rng(0).random((30, 3))
    return coords, PointCloud(coords, inds=np.arange(30))


@pytest.mark.xfail(strict=True, reason="returns rows of the slice, P1.4")
def test_closest_returns_rows_of_the_original_cloud(cloud):
    coords, pc = cloud

    assert np.all(pc[[5, 10, 20]].closest(coords[[20, 10]]).inds == [20, 10])


@pytest.mark.xfail(strict=True, reason="returns rows of the slice, P1.4")
def test_furthest_returns_rows_of_the_original_cloud(cloud):
    coords, pc = cloud
    targets = coords[[5, 20]]
    expected = pc.inds[[pc.index_of_furthest(p) for p in targets]]

    assert np.all(pc[:].furthest(targets).inds == expected)


@pytest.mark.xfail(strict=True, reason="doesn't infer the frame, P1.4")
def test_index_of_furthest_accepts_several_targets(cloud):
    coords, pc = cloud
    targets = coords[[20, 10]]
    expected = [pc.index_of_furthest(p) for p in targets]

    assert np.all(pc.index_of_furthest(targets) == expected)


@pytest.mark.xfail(strict=True, reason="docstrings say 'closest', P1.4")
def test_furthest_docstrings():
    for method in (PointCloud.furthest, PointCloud.index_of_furthest):
        summary = method.__doc__.strip().splitlines()[0]
        assert "furthest" in summary and "closest" not in summary
