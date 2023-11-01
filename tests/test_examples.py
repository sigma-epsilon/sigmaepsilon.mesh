import unittest

from sigmaepsilon.mesh.examples import compound_mesh


class TestExamples(unittest.TestCase):
    def test_compound_mesh(self):
        compound_mesh()


if __name__ == "__main__":
    unittest.main()
