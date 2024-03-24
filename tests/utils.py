from typing import Iterable

from testbook import testbook


def get_code_cells_of_testbook(tb: testbook) -> Iterable[int]:
    return [i for i, cell in enumerate(tb.cells) if cell.cell_type == "code"]
