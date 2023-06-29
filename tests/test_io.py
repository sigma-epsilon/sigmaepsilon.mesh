import unittest
import os
import numpy as np

from polymesh import PolyData, PointData
from polymesh.cells import T3, Q4, H8
from polymesh.trimesh import TriMesh
from polymesh.grid import Grid
from polymesh.space import StandardFrame
from neumann.logical import isclose


class TestIO(unittest.TestCase):
    def test_db_io(self):
        """
        1) Creates a compound volume, measures its volume, writes
        all the data to several parquet files, reads back in and measures
        the volume again. It is also checked if the topology has
        tight and zeroed indexing.
        2) Writes all pointdata and celldata to two files by merging
        block data.
        """
        A = StandardFrame(dim=3)
        tri = TriMesh(size=(100, 100), shape=(10, 10), frame=A)
        grid2d = Grid(size=(100, 100), shape=(10, 10), eshape="Q4", frame=A)
        grid3d = Grid(size=(100, 100, 100), shape=(8, 6, 2), eshape="H8", frame=A)

        mesh = PolyData(frame=A)
        mesh["tri", "T3"] = tri.move(np.array([0.0, 0.0, -50]))
        mesh["grids", "Q4"] = grid2d.move(np.array([0.0, 0.0, 150]))
        mesh["grids", "H8"] = grid3d

        mesh["tri", "T3"].pointdata["values"] = np.full(tri.coords().shape[0], 5.0)
        mesh["grids", "Q4"].pointdata["values"] = np.full(
            grid2d.coords().shape[0], 10.0
        )
        mesh["grids", "H8"].pointdata["values"] = np.full(
            grid3d.coords().shape[0], -5.0
        )

        volume = mesh.volume()

        mesh["tri", "T3"].pd.to_parquet("pdT3.parquet")
        mesh["grids", "Q4"].pd.to_parquet("pdQ4.parquet")
        mesh["grids", "H8"].pd.to_parquet("pdH8.parquet")
        mesh["tri", "T3"].cd.to_parquet("cdT3.parquet")
        mesh["grids", "Q4"].cd.to_parquet("cdQ4.parquet")
        mesh["grids", "H8"].cd.to_parquet("cdH8.parquet")
        paths = [
            "pdT3.parquet",
            "pdQ4.parquet",
            "pdH8.parquet",
            "cdT3.parquet",
            "cdQ4.parquet",
            "cdH8.parquet",
        ]

        mesh = PolyData(frame=A)
        pdT3 = PointData.from_parquet("pdT3.parquet")
        cdT3 = T3.from_parquet("cdT3.parquet")
        mesh["tri", "T3"] = PolyData(pdT3, cdT3, frame=A)
        pdQ4 = PointData.from_parquet("pdQ4.parquet")
        cdQ4 = Q4.from_parquet("cdQ4.parquet")
        mesh["grids", "Q4"] = PolyData(pdQ4, cdQ4, frame=A)
        pdH8 = PointData.from_parquet("pdH8.parquet")
        cdH8 = H8.from_parquet("cdH8.parquet")
        mesh["grids", "H8"] = PolyData(pdH8, cdH8, frame=A)

        self.assertTrue(isclose(volume, mesh.volume(), atol=1e-5, rtol=None))

        mesh.to_standard_form()
        t = mesh.topology()
        t0 = mesh.coords().shape[0]
        imin = np.min(t)
        t1 = np.max(t) - imin + 1
        self.assertEqual(t0, t1)
        self.assertEqual(imin, 0)

        ## PART 2
        mesh.to_parquet("mesh_pd.parquet", "mesh_cd.parquet")
        paths.extend(["mesh_pd.parquet", "mesh_cd.parquet"])
        mesh.to_dataframe()
        mesh["grids", "H8"].cd.to_dataframe()
        mesh["grids", "H8"].cd.to_akarray()
        mesh["grids", "H8"].cd.to_akrecord()

        for path in paths:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
