import matplotlib
import pytest


@pytest.fixture(autouse=True)
def set_mpl_backend():
    matplotlib.use("agg")
