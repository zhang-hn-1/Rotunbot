import unittest

from legged_gym.scripts.smoke_depth_local import _extract_backend, _extract_scene_mode


class DepthLocalSmokeTests(unittest.TestCase):
    def test_backend_is_explicitly_selected(self):
        backend, argv = _extract_backend(["--headless", "--depth-backend", "fallback"])
        self.assertEqual(backend, "fallback")
        self.assertEqual(argv, ["--headless"])

    def test_real_camera_backend_is_not_rewritten(self):
        backend, argv = _extract_backend(["--depth-backend", "isaacgym"])
        self.assertEqual(backend, "isaacgym")
        self.assertEqual(argv, [])

    def test_corridor_scene_is_explicit(self):
        scene, argv = _extract_scene_mode(["--scene-mode", "corridor", "--headless"])
        self.assertEqual(scene, "corridor")
        self.assertEqual(argv, ["--headless"])


if __name__ == "__main__":
    unittest.main()
