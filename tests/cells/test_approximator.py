import unittest
import doctest

import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
from sigmaepsilon.math.utils import to_range_1d
import sigmaepsilon.mesh
from sigmaepsilon.mesh.cells import H8, L2, L3, Q4, Q9
from sigmaepsilon.mesh.geometry import PolyCellGeometry1d


def load_tests(loader, tests, ignore):  # pragma: no cover
    tests.addTests(doctest.DocTestSuite(sigmaepsilon.mesh.cellapproximator))
    return tests


class TestLagrangianCellApproximator(SigmaEpsilonTestCase):
    def test_interpolator_H8(self):
        master_coordinates = H8.Geometry.master_coordinates()

        source_coordinates = master_coordinates / 2
        source_values = [1, 2, 3, 4, 5, 6, 7, 8]
        target_coordinates = master_coordinates
        approximator = H8.Geometry.approximator()
        approximator(
            source=source_coordinates, values=source_values, target=target_coordinates
        )

        source_coordinates = master_coordinates / 2
        source_values = [1, 2, 3, 4, 5, 6, 7, 8]
        target_coordinates = master_coordinates
        approximator = H8.Geometry.approximator(source_coordinates)
        self.assertFailsProperly(
            Exception,
            approximator,
            source=source_coordinates,
            values=source_values,
            target=target_coordinates,
        )

    def test_interpolator_H8_multi(self):
        approximator = H8.Geometry.approximator()
        master_coordinates = H8.Geometry.master_coordinates()

        source_coordinates = master_coordinates / 2
        target_coordinates = master_coordinates * 2

        source_values = np.random.rand(10, 2, 8)
        shape = approximator(
            source=source_coordinates,
            values=source_values,
            target=target_coordinates[:3],
        ).shape
        self.assertEqual(shape, (10, 2, 3))

        source_values = np.random.rand(8, 2, 10)
        shape = approximator(
            source=source_coordinates,
            values=source_values,
            target=target_coordinates[:3],
            axis=0,
        ).shape
        self.assertEqual(shape, (3, 2, 10))

    def test_interpolator_Q4_Q9(self):
        master_coordinates = Q9.Geometry.master_coordinates()
        source_coordinates = master_coordinates / 2
        source_values = [i + 1 for i in range(9)]
        target_coordinates = master_coordinates

        approximator = Q4.Geometry.approximator()
        approximator(
            source=source_coordinates, values=source_values, target=target_coordinates
        )

    def test_interpolator_Q9_Q4(self):
        master_coordinates = Q4.Geometry.master_coordinates()
        source_coordinates = master_coordinates / 2
        source_values = [i + 1 for i in range(4)]
        target_coordinates = master_coordinates

        approximator = Q9.Geometry.approximator()
        approximator(
            source=source_coordinates, values=source_values, target=target_coordinates
        )

    def test_interpolator_L3(self):
        master_coordinates = L3.Geometry.master_coordinates()
        source_coordinates = master_coordinates / 2
        source_values = [i + 1 for i in range(3)]
        target_coordinates = master_coordinates

        approximator = L3.Geometry.approximator()
        approximator(
            source=source_coordinates, values=source_values, target=target_coordinates
        )

    # TODO: this could be generalized and performed automatically for all cells
    def test_interpolator_L2_b2b(self):
        """
        This tests if the interpolation works in both directions and the original values can
        be retrieved by extrapolating on the interpolated values and switching source and target
        coordinates.
        """
        target_coordinates = to_range_1d(
            np.random.rand(2), source=[0, 1], target=[-1, 1]
        )
        source_values = np.random.rand(2)
        source_coordinates = L2.Geometry.master_coordinates()

        approximator = L2.Geometry.approximator()

        target_values = approximator(
            source=source_coordinates, values=source_values, target=target_coordinates
        )

        target_values_ = approximator(
            source=target_coordinates, values=target_values, target=source_coordinates
        )

        self.assertTrue(np.allclose(target_values_, source_values))

    def test_custom_approximator_1d(self):
        Custom1dCell: PolyCellGeometry1d = PolyCellGeometry1d.generate_class(
            number_of_nodes=4
        )

        NNODE = Custom1dCell.number_of_nodes

        target_coordinates = to_range_1d(
            np.random.rand(NNODE), source=[0, 1], target=[-1, 1]
        )
        source_values = np.random.rand(NNODE)

        source_coordinates = Custom1dCell.master_coordinates()

        approximator = Custom1dCell.approximator()

        target_values = approximator(
            source=source_coordinates, values=source_values, target=target_coordinates
        )

        target_values_ = approximator(
            source=target_coordinates, values=target_values, target=source_coordinates
        )

        self.assertTrue(np.allclose(target_values_, source_values))


if __name__ == "__main__":
    unittest.main()
