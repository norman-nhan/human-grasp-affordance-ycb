import tempfile
import unittest
from pathlib import Path

import numpy as np

from export_human_grasp_affordance import (
    derive_labels,
    load_obj,
    object_name_from_grasp,
    resolve_grasp_paths,
    sample_surface,
)


class ExportHumanGraspAffordanceTest(unittest.TestCase):
    def test_surface_sampling_is_deterministic_and_on_triangle(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        faces = np.array([[0, 1, 2]])
        first, normals = sample_surface(vertices, faces, 64, np.random.default_rng(4))
        second, _ = sample_surface(vertices, faces, 64, np.random.default_rng(4))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[:, :2] >= 0))
        self.assertTrue(np.all(first[:, 0] + first[:, 1] <= 1.0 + 1e-6))
        np.testing.assert_allclose(normals, np.array([[0, 0, 1]] * 64), atol=1e-7)

    def test_affordance_aggregates_union_frequency_and_proximity(self):
        object_points = np.array([[0, 0, 0], [0.01, 0, 0], [1, 0, 0]])
        hands = [np.array([[0, 0, 0]]), np.array([[0.01, 0, 0]])]
        distance, count, frequency, proximity = derive_labels(
            object_points, hands, threshold_m=0.002, sigma_m=0.01
        )
        np.testing.assert_allclose(distance, [0, 0, 0.99], atol=1e-6)
        np.testing.assert_array_equal(count, [1, 1, 0])
        np.testing.assert_allclose(frequency, [0.5, 0.5, 0])
        np.testing.assert_allclose(proximity[:2], [1, 1])
        self.assertEqual(float(proximity[2]), 0.0)

    def test_obj_loader_triangulates_quad(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.obj"
            path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
            vertices, faces = load_obj(path)
            self.assertEqual(vertices.shape, (4, 3))
            np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3]])

    def test_object_name_from_source_path(self):
        record = {
            "body": "/old/YCB-objects/025_mug/google_16k/nontextured_transformed.wrl"
        }
        self.assertEqual(object_name_from_grasp(record), "025_mug")

    def test_grasp_path_selection_accepts_basename_and_glob(self):
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            grasps = data_root / "grasps"
            grasps.mkdir()
            first = grasps / "obj_21_grasp_-1.pickle"
            second = grasps / "obj_21_grasp_-2.pickle"
            first.touch()
            second.touch()
            selected = resolve_grasp_paths(
                data_root, [str(grasps / "obj_21_grasp_-[12].pickle")]
            )
            self.assertEqual(selected, [first.resolve(), second.resolve()])
            self.assertEqual(
                resolve_grasp_paths(data_root, [first.name]), [first.resolve()]
            )


if __name__ == "__main__":
    unittest.main()
