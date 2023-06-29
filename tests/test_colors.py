import numpy as np
import unittest

from polymesh.utils.colors import rgb_to_hex, hex_to_rgb


class TestColors(unittest.TestCase):
    def test_colors(self):
        hex = rgb_to_hex((255, 255, 255))
        self.assertTrue(hex, "ffffff")

        rgb = hex_to_rgb("FF65BA")
        self.assertTrue(rgb, (255, 101, 186))

        rgb = tuple(np.random.randint(0, 255) for _ in range(3))
        self.assertTrue(hex_to_rgb(rgb_to_hex(rgb)), rgb)


if __name__ == "__main__":
    unittest.main()
