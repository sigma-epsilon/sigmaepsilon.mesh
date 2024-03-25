import pytest
from testbook import testbook

from .utils import get_code_cells_of_testbook


def test_cell_coords_1d():
    with testbook("tests/testbooks/test_cell_coords_1d.ipynb", execute=False) as tb:
        code_cell_indices = get_code_cells_of_testbook(tb)
        for index in code_cell_indices:
            tb.execute_cell(index)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
