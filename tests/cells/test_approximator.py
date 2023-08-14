import unittest
import doctest

import numpy as np

from sigmaepsilon.core.testing import SigmaEpsilonTestCase
import sigmaepsilon.mesh
from sigmaepsilon.mesh.cells import H8, L3, Q4, Q9


def load_tests(loader, tests, ignore):  # pragma: no cover
    tests.addTests(doctest.DocTestSuite(sigmaepsilon.mesh.cells.base.approximator))
    return tests


class TestLagrangianCellApproximator(SigmaEpsilonTestCase):
    def test_interpolator_H8(self):
        master_coordinates = H8.master_coordinates()

        source_coordinates = master_coordinates / 2
        source_values = [1, 2, 3, 4, 5, 6, 7, 8]
        target_coordinates = master_coordinates
        approximator = H8.approximator()
        approximator(
            source=source_coordinates, values=source_values, target=target_coordinates
        )

        source_coordinates = master_coordinates / 2
        source_values = [1, 2, 3, 4, 5, 6, 7, 8]
        target_coordinates = master_coordinates
        approximator = H8.approximator(source_coordinates)
        self.assertFailsProperly(
            Exception,
            approximator,
            source=source_coordinates,
            values=source_values,
            target=target_coordinates,
        )
        
    def test_interpolator_H8_multi(self):
        approximator = H8.approximator()
        master_coordinates = H8.master_coordinates()

        source_coordinates = master_coordinates / 2
        target_coordinates = master_coordinates * 2
        
        source_values = np.random.rand(10, 2, 8)
        shape = approximator(
            source=source_coordinates, 
            values=source_values, 
            target=target_coordinates[:3]
        ).shape
        self.assertEqual(shape, (10, 2, 3))
        
        source_values = np.random.rand(8, 2, 10)
        shape = approximator(
            source=source_coordinates, 
            values=source_values, 
            target=target_coordinates[:3],
            axis=0
        ).shape
        self.assertEqual(shape, (3, 2, 10))
        
    def test_interpolator_Q4_Q9(self):
        master_coordinates = Q9.master_coordinates()
        source_coordinates = master_coordinates / 2
        source_values = [i + 1 for i in range(9)]
        target_coordinates = master_coordinates

        approximator = Q4.approximator()
        approximator(source=source_coordinates, values=source_values, target=target_coordinates)
        
    def test_interpolator_Q9_Q4(self):
        master_coordinates = Q4.master_coordinates()
        source_coordinates = master_coordinates / 2
        source_values = [i + 1 for i in range(4)]
        target_coordinates = master_coordinates

        approximator = Q9.approximator()
        approximator(source=source_coordinates, values=source_values, target=target_coordinates)
        
    def test_interpolator_L3(self):
        master_coordinates = L3.master_coordinates()
        source_coordinates = master_coordinates / 2
        source_values = [i + 1 for i in range(3)]
        target_coordinates = master_coordinates

        approximator = L3.approximator()
        approximator(source=source_coordinates, values=source_values, target=target_coordinates)


if __name__ == "__main__":
    unittest.main()
