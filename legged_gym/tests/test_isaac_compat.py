import unittest

from legged_gym.navigation.isaac_compat import install_python_compat


class IsaacCompatTests(unittest.TestCase):
    def test_python_compat_exposes_legacy_aliases(self):
        install_python_compat()

        import distutils
        import numpy as np

        self.assertTrue(hasattr(distutils, "version"))
        self.assertIs(np.float, float)


if __name__ == "__main__":
    unittest.main()
