"""
Section 2.5 of the data model proposal: block metadata and configuration live
in ad hoc places.

Most tests pin today's behaviour, which P6 (block configuration) changes. The
test for ``pvplot`` removing the stored colour asserts the target behaviour of a
bug and is a strict ``xfail`` until P1.6.
"""

import numpy as np
import pytest

from sigmaepsilon.mesh import PolyData, LineData, PointData, CartesianFrame
from sigmaepsilon.mesh.cells import L2

from .conftest import BEAM_TOPOLOGY

pytestmark = pytest.mark.audit("2.5")

KEY = PolyData._pv_config_key_


def test_metadata_sits_in_the_dict_of_child_blocks(mixed_mesh: PolyData):
    mixed_mesh["Solids"]["material"] = "steel"

    assert mixed_mesh["Solids"]["material"] == "steel"
    assert list(mixed_mesh["Solids"].keys()) == ["material"]


def test_reading_a_typo_creates_an_empty_child_block(mixed_mesh: PolyData):
    _ = mixed_mesh["Solids"]["materal"]

    assert list(mixed_mesh["Solids"].keys()) == ["materal"]
    assert isinstance(mixed_mesh["Solids"]["materal"], PolyData)


def test_class_defaults_are_written_into_every_instance(mixed_mesh: PolyData):
    assert KEY == ("pv", "default")
    assert dict(mixed_mesh.config[KEY]) == {"show_edges": True}
    assert dict(mixed_mesh["Solids"].config[KEY]) == {"show_edges": True}


def test_value_set_on_parent_does_not_reach_children(mixed_mesh: PolyData):
    mixed_mesh.config[KEY]["color"] = "red"

    assert dict(mixed_mesh["Solids"]._get_config_(KEY)) == {"show_edges": True}


def test_lookup_replaces_whole_dicts_instead_of_merging(mixed_mesh: PolyData):
    mixed_mesh.config[KEY]["opacity"] = 0.5
    mixed_mesh["Solids"].config[KEY]["color"] = "blue"

    resolved = dict(mixed_mesh["Solids"]._get_config_(KEY))

    assert "opacity" not in resolved
    assert resolved == {"show_edges": True, "color": "blue"}


def test_linedata_defaults_live_in_instance_state():
    frame = CartesianFrame(dim=3)
    coords = np.zeros((11, 3))
    lines = LineData(
        PointData(coords=coords, frame=frame), L2(topo=BEAM_TOPOLOGY, frames=frame)
    )

    assert dict(lines.config[KEY]) == {
        "show_edges": True,
        "color": "k",
        "line_width": 10,
        "render_lines_as_tubes": True,
    }


@pytest.mark.xfail(strict=True, reason="pvplot pops from the stored config, P1.6")
def test_plotting_with_scalars_keeps_the_stored_colour(mixed_mesh: PolyData):
    pytest.importorskip("pyvista")
    block = mixed_mesh["Beams"]
    block.config[KEY]["color"] = "red"
    scalars = np.arange(len(mixed_mesh.pd), dtype=float)

    mixed_mesh.pvplot(
        notebook=False, scalars=scalars, return_plotter=True, off_screen=True
    )

    assert block.config[KEY]["color"] == "red"
