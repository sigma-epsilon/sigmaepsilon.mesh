# -*- coding: utf-8 -*-
import numpy as np
import unittest

from polymesh.grid import Grid


class TestHex(unittest.TestCase):
    def test_H8(self):
        def test_H8(Lx, Ly, Lz, nx, ny, nz):
            try:
                mesh = Grid(size=(Lx, Ly, Lz), shape=(nx, ny, nz), eshape="H8")
                assert np.isclose(mesh.volume(), Lx * Ly * Lz)
                return True
            except AssertionError:
                return False
            except Exception as e:
                raise e

        assert test_H8(1.0, 1.0, 1.0, 2, 2, 2)

    def test_H27(self):
        def test_H27(Lx, Ly, Lz, nx, ny, nz):
            try:
                mesh = Grid(size=(Lx, Ly, Lz), shape=(nx, ny, nz), eshape="H27")
                assert np.isclose(mesh.volume(), Lx * Ly * Lz)
                return True
            except AssertionError:
                return False
            except Exception as e:
                raise e

        assert test_H27(1.0, 1.0, 1.0, 2, 2, 2)


if __name__ == "__main__":
    unittest.main()
