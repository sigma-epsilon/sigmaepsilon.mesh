import matplotlib
import pytest
import asyncio
import os


@pytest.fixture(autouse=True)
def set_mpl_backend():
    matplotlib.use("agg")


# Windows specific settings
if os.name == "nt":

    @pytest.fixture(scope="session", autouse=True)
    def set_event_loop_policy():
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
