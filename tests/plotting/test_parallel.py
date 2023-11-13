import unittest
import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.mesh.plotting import parallel_mpl, aligned_parallel_mpl


class TestMplParallel(SigmaEpsilonTestCase):
    def test_parallel_mpl_plot_1(self):
        colors = np.random.rand(5, 3)
        labels = [str(i) for i in range(10)]
        values = np.random.rand(10, 5)
        parallel_mpl(
            values,
            labels=labels,
            padding=0.05,
            lw=0.2,
            colors=colors,
            title="Parallel Plot with Random Data",
            return_figure=True,
            bezier=False,
        )

    def test_parallel_mpl_plot_2(self):
        colors = np.random.rand(5, 3)
        values = {str(i): np.random.rand(5) for i in range(10)}
        parallel_mpl(
            values,
            padding=0.05,
            lw=0.2,
            colors=colors,
            title="Parallel Plot with Random Data",
            return_figure=False,
            bezier=False,
        )

    def test_parallel_mpl_plot_3(self):
        colors = np.random.rand(5, 3)
        labels = [str(i) for i in range(10)]
        values = [np.random.rand(5) for _ in range(10)]
        parallel_mpl(
            values,
            labels=labels,
            padding=0.05,
            lw=0.2,
            colors=colors,
            title="Parallel Plot with Random Data",
            return_figure=False,
            bezier=False,
        )

    def test_parallel_mpl_plot_4(self):
        colors = np.random.rand(5, 3)
        labels = [str(i) for i in range(10)]
        values = [np.random.rand(5) for _ in range(10)]
        parallel_mpl(
            values,
            labels=labels,
            padding=0.05,
            lw=0.2,
            colors=colors,
            title="Parallel Plot with Random Data",
            return_figure=False,
            bezier=True,
        )

    def test_aligned_parallel_mpl_plot_1(self):
        labels = ["a", "b", "c"]
        values = np.array([np.random.rand(10) for _ in labels]).T
        datapos = np.linspace(-1, 1, 10)

        aligned_parallel_mpl(
            values,
            datapos,
            labels=labels,
            yticks=[-1, 1],
            return_figure=False,
            slider=True,
        )

        aligned_parallel_mpl(
            values,
            datapos,
            labels=labels,
            yticks=[-1, 1],
            return_figure=True,
            y0=0.0,
            slider=True,
        )

        aligned_parallel_mpl(
            values,
            datapos,
            yticks=[-1, 1],
            return_figure=True,
            y0=0.0,
            slider=True,
        )

        values = {label: np.random.rand(10) for label in labels}

        aligned_parallel_mpl(
            values,
            datapos,
            yticks=[-1, 1],
            return_figure=True,
            y0=0.0,
            slider=True,
        )


if __name__ == "__main__":
    unittest.main()
