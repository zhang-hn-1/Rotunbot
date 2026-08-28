"""Freeze a validated V49 velocity-tracker release without altering its source run."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid


RELEASE_TASK = "rotunbot_vel_sru50_v49"
VALIDATION_DIR = "final_v49_validation_slope027"
SOURCE_SNAPSHOT = (
    "legged_gym/envs/rotunbot/vel_tracking/rotunbot_vel.py",
    "legged_gym/envs/rotunbot/vel_tracking/rotunbot_vel_config.py",
    "legged_gym/envs/__init__.py",
    "legged_gym/scripts/scan_vel_reachable_domain.py",
    "legged_gym/scripts/evaluate_vel_tracking_release.py",
    "legged_gym/scripts/check_vel_release_gate.py",
    "legged_gym/scripts/validate_vel_sru50_v49_gpu23.sh",
    "legged_gym/scripts/freeze_vel_sru50_v49_release.py",
    "legged_gym/tests/test_rotunbot_velocity_tracking.py",
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_run", help="V48 source run containing model_<N>.pt")
    parser.add_argument("checkpoint", type=int)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Frozen release directory (must not already exist)",
    )
    return parser.parse_args()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _git_text(root, *args):
    result = subprocess.run(
        ("git",) + args,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _validate_results(validation_root):
    grid_path = validation_root / "reachable_domain" / "reachable_domain_summary.json"
    if not grid_path.is_file():
        raise FileNotFoundError(grid_path)
    grid = _read_json(grid_path)
    if grid.get("verdict") != "PASS":
        raise RuntimeError("Reachable-domain grid did not pass")

    release_paths = sorted(validation_root.glob("release_seed_*/release_summary.json"))
    if len(release_paths) != 3:
        raise RuntimeError(
            "Exactly three held-out release summaries are required; "
            f"found {len(release_paths)}"
        )
    releases = []
    for path in release_paths:
        summary = _read_json(path)
        checks = summary.get("checks", {})
        if summary.get("verdict") != "PASS" or not checks or not all(checks.values()):
            raise RuntimeError(f"Release evaluation did not pass: {path}")
        releases.append(summary)
    return grid_path, grid, release_paths, releases


def _copy_source_snapshot(root, staging):
    for relative in SOURCE_SNAPSHOT:
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        target = staging / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _file_inventory(directory):
    files = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        relative = path.relative_to(directory).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return files


def main():
    args = _parse_args()
    root = Path(__file__).resolve().parents[2]
    source_run = Path(args.source_run)
    if not source_run.is_absolute():
        source_run = (root / source_run).resolve()
    checkpoint = int(args.checkpoint)
    source_model = source_run / f"model_{checkpoint}.pt"
    validation_root = source_run / VALIDATION_DIR / f"checkpoint_{checkpoint}"
    if not source_model.is_file():
        raise FileNotFoundError(source_model)

    grid_path, grid, release_paths, releases = _validate_results(validation_root)
    if args.output_dir:
        output = Path(args.output_dir)
        if not output.is_absolute():
            output = (root / output).resolve()
    else:
        output = (
            root
            / "logs"
            / "rotunbot_vel_sru50_v49_release"
            / f"frozen_checkpoint_{checkpoint}_20260827"
        )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen release: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        frozen_model = staging / "model_frozen.pt"
        shutil.copy2(source_model, frozen_model)
        shutil.copytree(validation_root, staging / "validation")
        _copy_source_snapshot(root, staging)

        readme = staging / "FROZEN_RELEASE.md"
        readme.write_text(
            "# Frozen SRU-compatible velocity tracker V49\n\n"
            "This directory is an immutable evaluation release. Do not resume PPO "
            "training from `model_frozen.pt`. The original V48 training run remains "
            "unchanged at the source path recorded in `manifest.json`.\n\n"
            "Timing: 200 Hz physics, 50 Hz low-level policy, 5 Hz upper command.\n\n"
            "Validated non-extreme domain: |v| <= 0.13 m/s and "
            "|omega| <= 0.27 |v| after the measured low-speed steering-authority "
            "fade from 0.08 to 0.10 m/s. In-place turning is excluded.\n",
            encoding="utf-8",
        )

        release_metrics = []
        for path, summary in zip(release_paths, releases):
            release_metrics.append(
                {
                    "seed": summary.get("seed", path.parent.name),
                    "verdict": summary["verdict"],
                    "step": summary["step"],
                    "sine": summary["sine"],
                    "random_continuous": summary["random_continuous"],
                    "checks": summary["checks"],
                }
            )

        manifest = {
            "schema_version": 1,
            "release_task": RELEASE_TASK,
            "trained_task": "rotunbot_vel_sru50_v48",
            "checkpoint": checkpoint,
            "frozen_parameters": True,
            "resume_training_permitted": False,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_run": str(source_run),
            "source_model": str(source_model),
            "source_model_sha256": _sha256(source_model),
            "timing_hz": {
                "physics": 200,
                "low_level_policy": 50,
                "upper_command": 5,
            },
            "command_contract": {
                "direct_v_w_tracking": True,
                "hidden_command_governor": False,
                "max_abs_v_mps": 0.13,
                "max_abs_w_radps": 0.10,
                "effective_yaw_slope_abs_w_over_abs_v": 0.27,
                "minimum_turn_radius_m": 3.148148148148148,
                "turn_authority_start_speed_mps": 0.08,
                "turn_authority_full_speed_mps": 0.10,
                "in_place_turning": False,
            },
            "controller": {
                "angular_feedback_gain": 0.40,
                "angular_feedback_action_limit": 0.30,
                "angular_rate_feedforward_time_s": 0.65,
                "angular_rate_feedforward_action_limit": 0.12,
            },
            "reward_contract": {
                "tracking_lin_vel_scale": 12.0,
                "tracking_ang_vel_scale": 36.0,
                "angular_tracking_error_scale": -6.0,
                "curvature_tracking_scale": 3.0,
                "curvature_is_auxiliary": True,
            },
            "validation": {
                "reachable_domain_summary": grid,
                "release_evaluations": release_metrics,
                "thresholds": releases[0].get("thresholds", {}),
                "strict_gate": "PASS",
            },
            "git": {
                "commit": _git_text(root, "rev-parse", "HEAD"),
                "status_short": _git_text(root, "status", "--short"),
            },
        }
        manifest["files"] = _file_inventory(staging)
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        os.chmod(frozen_model, 0o444)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Frozen release: {output}")
    print(f"Model SHA256: {_sha256(output / 'model_frozen.pt')}")
    print(f"Grid verdict: {grid['verdict']}")
    print(f"Release verdicts: {[summary['verdict'] for summary in releases]}")


if __name__ == "__main__":
    main()
