"""Physical audit for the formal Phase-D IsaacGym IMAGE_DEPTH camera."""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np


def _framework_args(values):
    from legged_gym.utils import get_args

    original = list(os.sys.argv)
    os.sys.argv = [original[0]] + list(values)
    try:
        return get_args()
    finally:
        os.sys.argv = original


def _wall(start, end):
    return ((float(start[0]), float(start[1])), (float(end[0]), float(end[1])))


def _cases():
    return {
        "A_empty": (),
        "B_wall_2m": (_wall((2.0, -2.0), (2.0, 2.0)),),
        "C_wall_1m": (_wall((1.0, -2.0), (1.0, 2.0)),),
        "D_left": (_wall((1.5, -1.8), (1.5, -0.35)),),
        "E_right": (_wall((1.5, 0.35), (1.5, 1.8)),),
    }


def _make_env(framework_args, segments):
    import legged_gym.envs  # noqa: F401
    from legged_gym.utils import task_registry

    args = _framework_args(framework_args)
    args.task = "rotunbot_sru_visual_corridor_v1"
    cfg, _ = task_registry.get_cfgs(args.task)
    cfg.env.num_envs = 1
    cfg.enable_camera_sensors_in_headless = True
    cfg.camera.depth_backend = "isaacgym"
    cfg.camera.add_noise = False
    cfg.camera.position = (0.42, 0.0, 0.0)
    cfg.camera.rotation = (0.0, 0.0, 0.0, 1.0)
    cfg.corridor_wall_segments = ()
    cfg.corridor_explicit_wall_segments = tuple(segments)
    cfg.direct_obstacle_aabbs = ()
    cfg.commands.v1_goal_curriculum_enabled = False
    cfg.commands.v1_performance_curriculum_enabled = False
    cfg.init_state.randomize_initial_velocity = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False
    return task_registry.make_env(args.task, args=args, env_cfg=cfg)


def _capture(env):
    import torch

    env.reset()
    # compute_observations performs the production capture path and sets the
    # authoritative backend after render/fetch/access synchronization.
    env.compute_observations()
    requested = str(getattr(env, "depth_backend_requested", "unknown"))
    actual = str(getattr(env, "depth_backend_actual", "unknown"))
    if requested != "isaacgym" or actual != "isaacgym":
        raise RuntimeError("formal audit requires isaacgym/isaacgym, got %s/%s" % (requested, actual))
    raw = getattr(env, "_last_depth_raw", None)
    if raw is None:
        raw = torch.stack(env._camera_depth_tensors, dim=0).detach().clone()
    normalized = env.depth_observation.detach().clone()
    return raw[0].float().cpu().numpy(), normalized[0].float().cpu().numpy(), env.depth_capture_metadata()


def _close(env):
    if env is None:
        return
    if getattr(env, "viewer", None) is not None:
        env.gym.destroy_viewer(env.viewer)
    if getattr(env, "sim", None) is not None:
        env.gym.destroy_sim(env.sim)


def _save_case(output, name, raw, normalized):
    samples = output / "depth_samples"
    samples.mkdir(parents=True, exist_ok=True)
    np.save(samples / (name + "_raw.npy"), raw)
    np.save(samples / (name + "_normalized.npy"), normalized)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(8, 2.5))
        axis.imshow(normalized, vmin=0.0, vmax=1.0, aspect="auto", cmap="gray")
        axis.set_title(name + " normalized IMAGE_DEPTH")
        axis.set_xlabel("camera column")
        axis.set_ylabel("camera row")
        figure.tight_layout()
        figure.savefig(samples / (name + ".png"), dpi=150)
        plt.close(figure)
    except Exception:
        # NPY artifacts remain authoritative when plotting dependencies are not
        # available in a headless validation image.
        pass


def _case_metrics(name, raw, normalized, camera_position, near, far):
    finite = np.isfinite(normalized)
    height, width = normalized.shape
    center = float(normalized[height // 2, width // 2])
    raw_center = float(raw[height // 2, width // 2])
    finite_values = normalized[finite]
    camera_x = float(camera_position[0])
    expected = None
    if name.endswith("wall_1m"):
        expected = 1.0 - camera_x
    elif name.endswith("wall_2m"):
        expected = 2.0 - camera_x
    expected_norm = None if expected is None else float(np.clip((expected - near) / (far - near), 0.0, 1.0))
    return {
        "case": name,
        "raw_shape": list(raw.shape),
        "normalized_shape": list(normalized.shape),
        "finite_ratio": float(finite.mean()),
        "raw_center": raw_center,
        "center_normalized": center,
        "normalized_min": float(finite_values.min()) if finite_values.size else None,
        "normalized_max": float(finite_values.max()) if finite_values.size else None,
        "normalized_mean": float(finite_values.mean()) if finite_values.size else None,
        "normalized_std": float(finite_values.std()) if finite_values.size else None,
        "predicted_camera_axis_distance_m": expected,
        "predicted_center_normalized": expected_norm,
        "center_error_normalized": None if expected_norm is None else abs(center - expected_norm),
    }


def validate_cases(measurements):
    by_name = {item["case"]: item for item in measurements}
    failures = []
    for item in measurements:
        if item["finite_ratio"] < 1.0:
            failures.append(item["case"] + ":nonfinite")
    if "A_empty" in by_name and by_name["A_empty"]["center_normalized"] < 0.99:
        failures.append("A_empty:not_far_plane")
    if "B_wall_2m" in by_name and "C_wall_1m" in by_name:
        if not by_name["C_wall_1m"]["center_normalized"] < by_name["B_wall_2m"]["center_normalized"]:
            failures.append("C_vs_B:not_monotonic")
    for name in ("B_wall_2m", "C_wall_1m"):
        error = by_name.get(name, {}).get("center_error_normalized")
        if error is not None and error > 0.08:
            failures.append(name + ":geometric_error")
    if "D_left" in by_name and "E_right" in by_name:
        # A mirrored side obstacle can subtend a different number of pixels at
        # the image edge.  Validate response presence and mirrored centroids,
        # rather than requiring an invalid pixel-for-pixel equality.
        left_columns = by_name["D_left"].get("response_columns", [])
        right_columns = by_name["E_right"].get("response_columns", [])
        if not left_columns or not right_columns:
            failures.append("D_E:missing_side_response")
        else:
            width = 32
            centroid_sum = (
                float(by_name["D_left"].get("response_centroid_column", 0.0))
                + float(by_name["E_right"].get("response_centroid_column", 0.0))
            )
            if abs(centroid_sum - (width - 1)) > 2.0:
                failures.append("D_E:not_mirror_consistent")
    return {"pass": not failures, "failures": failures}


def _run_one_case(framework_args, name, segments, output):
    env = None
    try:
        env, cfg = _make_env(framework_args, segments)
        raw, normalized, metadata = _capture(env)
        metric = _case_metrics(name, raw, normalized, cfg.camera.position, float(cfg.camera.near_plane), float(cfg.camera.far_plane))
        metric["normalized_image"] = normalized.tolist()
        response = 1.0 - normalized.mean(axis=0)
        metric["response_columns"] = [int(index) for index in np.flatnonzero(response > 0.05)]
        metric["response_centroid_column"] = float((np.arange(normalized.shape[1]) * response).sum() / max(response.sum(), 1.0e-9))
        _save_case(output, name, raw, normalized)
        return {"metric": metric, "camera": {
            "position_m": list(cfg.camera.position),
            "rotation_quat": list(cfg.camera.rotation),
            "horizontal_fov_deg": float(cfg.camera.horizontal_fov),
            "near_plane_m": float(cfg.camera.near_plane),
            "far_plane_m": float(cfg.camera.far_plane),
            "resolution": [int(cfg.camera.height), int(cfg.camera.width)],
            "backend_requested": str(env.depth_backend_requested),
            "backend_actual": str(env.depth_backend_actual),
            "capture_metadata": metadata,
        }}
    except Exception as exc:
        return {"error": {"case": name, "error": repr(exc), "traceback": traceback.format_exc()}}
    finally:
        _close(env)


def _worker_main(argv):
    output, name = argv[0], argv[1]
    segments = _cases()[name]
    result = _run_one_case([], name, segments, Path(output))
    Path(output, name + ".json").write_text(json.dumps(result, sort_keys=True), encoding="utf-8")


def run_audit(framework_args=(), output="logs/phase_d/depth_camera_audit"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    measurements = []
    errors = []
    camera_metadata = None
    script = str(Path(__file__).resolve())
    for name in _cases():
        child_output = output / ("case_" + name)
        child_output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, script, "--worker", str(child_output), name]
        env = dict(os.environ)
        if framework_args:
            command.extend(framework_args)
        completed = subprocess.run(command, env=env, cwd=str(Path(__file__).resolve().parents[2]), text=True, capture_output=True)
        result_path = child_output / (name + ".json")
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = {"error": {"case": name, "error": completed.stderr[-4000:] or "worker failed"}}
        if "error" in result:
            errors.append(result["error"])
        else:
            metric = result["metric"]
            measurements.append(metric)
            camera_metadata = result["camera"]
    if errors:
        gate = {"pass": False, "failures": [item["case"] + ":process_failure" for item in errors]}
    else:
        gate = validate_cases(measurements)
    report = {
        "DEPTH_CAMERA_GATE": "PASS" if gate["pass"] else "FAIL",
        "gate": gate,
        "camera": camera_metadata,
        "measurements": measurements,
        "errors": errors,
    }
    (output / "depth_camera_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    markdown = ["# Phase-D Depth Camera Audit", "", "`DEPTH_CAMERA_GATE = %s`" % report["DEPTH_CAMERA_GATE"], ""]
    if camera_metadata:
        markdown.extend(["## Camera", "", "```json", json.dumps(camera_metadata, indent=2, sort_keys=True), "```", ""])
    markdown.extend(["## Gate", "", "```json", json.dumps(gate, indent=2, sort_keys=True), "```", ""])
    (output / "depth_camera_audit.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv=None):
    values = sys.argv[1:] if argv is None else list(argv)
    if "--worker" in values:
        index = values.index("--worker")
        output = values[index + 1]
        name = values[index + 2]
        _worker_main((output, name))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="logs/phase_d/depth_camera_audit")
    stage, remaining = parser.parse_known_args(values)
    return run_audit(remaining, stage.output)


if __name__ == "__main__":
    main()
