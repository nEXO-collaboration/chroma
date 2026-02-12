import numpy as np
import unittest

from chroma.gpu.tools import to_float3, to_uint3
from chroma.geometry import Mesh


def _as_plain_3col(arr, dtype):
    # Convert structured vec arrays back to plain Nx3 for value comparison.
    return np.array([tuple(x) for x in arr], dtype=dtype).reshape(-1, 3)


class TestNumpyCompatRegression(unittest.TestCase):
    def test_to_uint3_roundtrip_values_and_layout(self):
        src = np.array([[1, 2, 3], [10, 20, 30]], dtype=np.int64)
        out = to_uint3(src)
        back = _as_plain_3col(out, np.uint32)
        np.testing.assert_array_equal(back, src.astype(np.uint32))

    def test_to_float3_roundtrip_values_and_layout(self):
        src = np.array([[1.5, 2.25, 3.75], [10.0, 20.5, 30.25]], dtype=np.float64)
        out = to_float3(src)
        back = _as_plain_3col(out, np.float32)
        np.testing.assert_allclose(back, src.astype(np.float32), rtol=0, atol=0)

    def test_to_vec3_accepts_flat_multiple_of_three(self):
        src = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
        out = to_uint3(src)
        back = _as_plain_3col(out, np.uint32)
        np.testing.assert_array_equal(back, src.reshape(-1, 3).astype(np.uint32))

    def test_to_vec3_rejects_bad_shapes(self):
        bad = np.zeros((2, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            to_float3(bad)
        with self.assertRaises(ValueError):
            to_uint3(bad)

    def test_mesh_remove_null_triangles_keeps_non_degenerate_and_drops_degenerate(self):
        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
        )
        tris = np.array(
            [
                [0, 1, 2],  # valid
                [1, 1, 2],  # degenerate
                [0, 2, 3],  # valid
                [3, 3, 3],  # degenerate
            ],
            dtype=np.int32,
        )
        m = Mesh(verts, tris, remove_duplicate_vertices=False, remove_null_triangles=True)
        np.testing.assert_array_equal(m.triangles, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
