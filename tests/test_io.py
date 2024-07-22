import unittest

from sigmaepsilon.mesh import PolyData


class TestImports(unittest.TestCase):
    def test_import_from_stl(self):
        stl_file_path = "tests/files/bike_stem.stl"
        PolyData.from_stl(stl_file_path, clean=True, repair=True, verbose=True)
        PolyData.from_stl(stl_file_path, clean=False, repair=False, verbose=True)
        

if __name__ == "__main__":
    unittest.main()