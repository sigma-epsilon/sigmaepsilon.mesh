import numpy as np
import unittest

from neumann.linalg import Vector, linspace
from polymesh import PointCloud, CartesianFrame
from polymesh.triang import triangulate
from polymesh.space import StandardFrame
from polymesh.utils.utils import center_of_points


class TestLinalg(unittest.TestCase):
    def test_pointcloud_basic(self):
        def test_pointcloud_basic():
            coords, *_ = triangulate(size=(800, 600), shape=(10, 10))
            coords = PointCloud(coords)
            coords.center()
            coords.centralize()
            d = np.array([1.0, 0.0, 0.0])
            coords.rotate("Body", [0, 0, np.pi / 2], "XYZ").move(d)
            return np.all(np.isclose(coords.center(), d))

        assert test_pointcloud_basic()

    def test_pointcloud_path_1(self):
        def test_pointcloud_path_1():
            """
            A triangulation travels a self-closing cycle and should return to
            itself (DCM matrix of the frame must be the identity matrix).
            """
            coords, *_ = triangulate(size=(800, 600), shape=(10, 10))
            coords = PointCloud(coords)
            coords.centralize()
            old = coords.show()
            d = np.array([1.0, 0.0, 0.0])
            r = "Body", [0, 0, np.pi / 2], "XYZ"
            coords.move(d, coords.frame).rotate(*r).move(d, coords.frame).rotate(
                *r
            ).move(d, coords.frame).rotate(*r).move(d, coords.frame).rotate(*r)
            new = coords.show()
            hyp_1 = np.all(np.isclose(old, new))
            hyp_2 = np.all(np.isclose(np.eye(3), coords.frame.dcm()))
            return hyp_1 & hyp_2

        assert test_pointcloud_path_1()

    def test_pointcloud_1(self):
        c = np.array([[0, 0, 0], [0, 0, 1.0], [0, 0, 0]])
        COORD = PointCloud(c, inds=np.array([0, 1, 2, 3]))
        COORD[:, :].inds
        arr1 = COORD[1:]
        arr1.inds
        arr2 = arr1[1:]
        arr2.inds
        COORD.to_numpy()
        COORD.show() @ np.eye(3)
        coords, topo, _ = triangulate(size=(800, 600), shape=(10, 10))
        coords = PointCloud(coords)
        coords.center()
        coords.centralize()
        coords.center()
        frameA = CartesianFrame(axes=np.eye(3), origo=np.array([-500.0, 0.0, 0.0]))
        frameB = CartesianFrame(axes=np.eye(3), origo=np.array([+500.0, 0.0, 0.0]))
        frameA.origo
        frameB.origo
        coords.center()
        coords.center(frameA)
        coords.center(frameB)

    def test_pointcloud_2(self):
        Lx, Ly, Lz = 300, 300, 300
        points_per_edge = 3
        mesh_size = Lx / (points_per_edge - 1)
        points = []
        nTotalPoints = 0  # node counter

        # corners
        corner_coords = [
            [-Lx / 2, -Ly / 2, -Lz / 2],
            [Lx / 2, -Ly / 2, -Lz / 2],
            [Lx / 2, Ly / 2, -Lz / 2],
            [-Lx / 2, Ly / 2, -Lz / 2],
            [-Lx / 2, -Ly / 2, Lz / 2],
            [Lx / 2, -Ly / 2, Lz / 2],
            [Lx / 2, Ly / 2, Lz / 2],
            [-Lx / 2, Ly / 2, Lz / 2],
        ]
        corner_coords = np.array(corner_coords)
        points.append(corner_coords)
        nTotalPoints += len(corner_coords)

        # populate edges
        nodes_of_edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        edge_coords = []
        N = points_per_edge + 2
        for nodes in nodes_of_edges:
            p0 = corner_coords[nodes[0]]
            p1 = corner_coords[nodes[1]]
            edge_coords.append(linspace(p0, p1, N)[1:-1])
        edge_coords = np.vstack(edge_coords)
        points.append(edge_coords)
        nTotalPoints += len(edge_coords)

        # faces
        corners_of_faces = {
            "front": [1, 2, 6, 5],
            "back": [0, 3, 7, 4],
            "left": [2, 3, 7, 6],
            "right": [0, 1, 5, 4],
            "bottom": [0, 1, 2, 3],
            "top": [4, 5, 6, 7],
        }
        edges_of_faces = {
            "front": [1, 5, 9, 10],
            "back": [3, 7, 8, 11],
            "right": [0, 9, 4, 8],
            "left": [2, 6, 10, 11],
            "bottom": [0, 1, 2, 3],
            "top": [4, 5, 6, 7],
        }

        # center of face
        def cof(id):
            return center_of_points(corner_coords[corners_of_faces[id]])

        # face frames
        frames = {}
        frames["front"] = StandardFrame(dim=3, origo=cof("front"))
        rot90z = "Body", [0, 0, np.pi / 2], "XYZ"
        frames["left"] = frames["front"].fork(*rot90z).move(cof("left") - cof("front"))
        frames["back"] = frames["left"].fork(*rot90z).move(cof("back") - cof("left"))
        frames["right"] = frames["back"].fork(*rot90z).move(cof("right") - cof("back"))
        rot_front_top = "Body", [0, -np.pi / 2, 0], "XYZ"
        frames["top"] = (
            frames["front"].fork(*rot_front_top).move(cof("top") - cof("front"))
        )
        rot180y = "Body", [0, np.pi, 0], "XYZ"
        frames["bottom"] = frames["top"].fork(*rot180y).move(cof("bottom") - cof("top"))

    def test_frame_1(self):
        A = CartesianFrame()
        B = A.orient_new("Body", [0, 0, 45 * np.pi / 180], "XYZ")
        B.move(Vector(np.array([1.0, 0, 0]), frame=B))
        B.move(-Vector(np.array([np.sqrt(2) / 2, 0, 0])))
        B.move(-Vector(np.array([0, np.sqrt(2) / 2, 0])))

    def test_frame_2(self):
        A = CartesianFrame()
        B = A.orient_new("Body", [0, 0, 30 * np.pi / 180], "XYZ")
        C = B.orient_new("Body", [0, 0, 30 * np.pi / 180], "XYZ")
        A = CartesianFrame()
        B = A.orient_new("Body", [0, 0, 30 * np.pi / 180], "XYZ")
        C = B.orient_new("Body", [0, 0, 30 * np.pi / 180], "XYZ")
        A = CartesianFrame()
        v = Vector(np.array([1.0, 0.0, 0.0]), frame=A)
        B = A.fork("Body", [0, 0, 45 * np.pi / 180], "XYZ").move(v)


if __name__ == "__main__":
    unittest.main()
