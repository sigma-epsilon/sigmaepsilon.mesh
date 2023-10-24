# -*- coding: utf-8 -*-
import unittest, doctest

from numpy import ndarray

import sigmaepsilon.mesh.domains.section
from sigmaepsilon.mesh.domains.section import LineSection, get_section
from sigmaepsilon.mesh import TriMesh, CartesianFrame, PolyData


def load_tests(loader, tests, ignore):  # pragma: no cover
    tests.addTests(doctest.DocTestSuite(sigmaepsilon.mesh.domains.section))
    return tests


class TestSection(unittest.TestCase):
    def test_initialization(self):
        LineSection(get_section("CHS", d=1.0, t=0.1, n=16))
        LineSection("CHS", d=1.0, t=0.1, n=16)
        LineSection(shape="CHS", d=1.0, t=0.1, n=16)

    def test_CHS(self):
        LineSection("CHS", d=1.0, t=0.1, n=16)

    def test_RS(self):
        LineSection("RS", d=1.0, b=1.0)

    def test_RHS(self):
        LineSection("RHS", d=100, b=50, t=6, r_out=9, n_r=8)

    def test_I(self):
        LineSection("I", d=203, b=133, t_f=7.8, t_w=5.8, r=8.9, n_r=16)

    def test_TFI(self):
        LineSection(
            "TFI", d=588, b=191, t_f=27.2, t_w=15.2, r_r=17.8, r_f=8.9, alpha=8, n_r=16
        )

    def test_PFC(self):
        LineSection("PFC", d=250, b=90, t_f=15, t_w=8, r=12, n_r=8)

    def test_TFC(self):
        LineSection(
            "TFC",
            d=10,
            b=3.5,
            t_f=0.575,
            t_w=0.475,
            r_r=0.575,
            r_f=0.4,
            alpha=8,
            n_r=16,
        )

    def test_T(self):
        LineSection("T", d=200, b=100, t_f=12, t_w=6, r=8, n_r=8)

    def test_section_properties(self):
        section = LineSection("CHS", d=1.0, t=0.5, n=16)
        self.assertIsInstance(section.calculate_section_properties(), dict)
        self.assertIsInstance(section.get_section_properties(), dict)
        self.assertIsInstance(section.A, float)
        self.assertIsInstance(section.Ix, float)
        self.assertIsInstance(section.Iy, float)
        self.assertIsInstance(section.Iz, float)
        self.assertIsInstance(section.geometric_properties, dict)
        self.assertIsInstance(section.section_properties, dict)
        self.assertIsInstance(section.trimesh(), TriMesh)
        self.assertIsInstance(section.trimesh(subdivide=True), TriMesh)
        self.assertIsInstance(section.coords(), ndarray)
        self.assertIsInstance(section.topology(), ndarray)

    def test_extrude(self):
        frame = CartesianFrame(dim=3)
        section = LineSection("CHS", d=1.0, t=0.5, n=16)
        self.assertIsInstance(section.extrude(length=1.0, N=2, frame=frame), PolyData)

    def test_coverage_plus(self):
        """Extra tests for coverage."""
        section = LineSection("CHS", d=1.0, t=0.5, n=16)
        self.assertIsInstance(section.get_section_properties(), dict)


if __name__ == "__main__":
    unittest.main()
