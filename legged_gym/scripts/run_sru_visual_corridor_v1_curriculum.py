"""Run V1 training and independent curriculum evaluation in isolated processes."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from legged_gym.navigation.v1_curriculum import (
    V1_CURRICULUM_LEVELS,
    V1PerformanceCurriculum,
)
from legged_gym.navigation.v1_evaluation import curriculum_gate
from legged_gym.scripts.train_sru_visual_corridor_v1 import (
    _write_history_row,
    curriculum_history_row,
)


LOG_DIRECTORY_RE = re.compile(r"TensorBoard log directory:\s*(.+?)\s*$", re.MULTILINE)


def build_training_command(
    python_executable,
    repo_root,
    resume_path,
    parent_checkpoint,
    curriculum_state,
    iterations,
    num_envs,
    framework_args=(),
):
    """Build one model-only training chunk; evaluation is never in this process."""
    script = Path(repo_root) / "legged_gym/scripts/train_sru_visual_corridor_v1.py"
    command = [
        str(python_executable),
        str(script),
        "--iterations", str(int(iterations)),
        "--num_envs", str(int(num_envs)),
        "--resume_path", str(resume_path),
        "--parent_checkpoint", str(parent_checkpoint),
        "--curriculum-state", str(curriculum_state),
        "--disable_camera_noise",
    ]
    return command + [str(value) for value in framework_args]


def build_evaluation_command(
    python_executable,
    repo_root,
    checkpoint,
    current_distance,
    next_distance,
    episodes,
    seed,
    output_dir,
    framework_args=(),
    num_envs=16,
):
    """Build one independent current/next 30+30 evaluation command."""
    script = Path(repo_root) / "legged_gym/scripts/eval_sru_visual_corridor_v1.py"
    command = [
        str(python_executable),
        str(script),
        "--checkpoint", str(checkpoint),
        "--current-distance", str(float(current_distance)),
        "--next-distance", str(float(next_distance)),
        "--episodes", str(int(episodes)),
        "--seed", str(int(seed)),
        "--num_envs", str(int(num_envs)),
        "--output-dir", str(output_dir),
    ]
    return command + [str(value) for value in framework_args]


def parse_log_directory(output):
    """Extract the training run directory printed by TaskRegistry."""
    matches = LOG_DIRECTORY_RE.findall(str(output))
    if not matches:
        raise ValueError("training output did not contain a TensorBoard log directory")
    return matches[-1].strip()


def _run(command, cwd, env):
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            "subprocess failed with exit code {}: {}".format(
                completed.returncode, " ".join(command)
            )
        )
    return completed.stdout


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _load_state(path, seed):
    if Path(path).is_file():
        return V1PerformanceCurriculum.from_dict(json.loads(Path(path).read_text()))
    return V1PerformanceCurriculum(seed=seed)


def _record_external_evaluation(curriculum, iteration, summary):
    current = summary["targets"]["current"]
    following = summary["targets"]["next"]
    gate = curriculum_gate(current, following)
    result = curriculum.record_evaluation(
        iteration=iteration,
        frontier_success=following["success_count"],
        replay_success=current["success_count"],
        collision_count=max(current["collision_count"], following["collision_count"]),
        rate_violation_count=max(
            current["rate_violation_count"], following["rate_violation_count"]
        ),
        domain_violation_count=max(
            current["feasible_domain_violation_count"],
            following["feasible_domain_violation_count"],
        ),
        hidden_projection_jump_count=max(
            current["hidden_projection_jump_count"],
            following["hidden_projection_jump_count"],
        ),
    )
    return result, gate, current, following


def run_curriculum(args, framework_args=()):
    """Train/evaluate chunks until 6 m or the stage budget is exhausted."""
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "curriculum_state.json"
    curriculum = _load_state(state_path, args.seed)
    _write_json(state_path, curriculum.to_dict())
    parent_checkpoint = str(Path(args.parent_checkpoint or args.resume_path).resolve())
    checkpoint = str(Path(args.resume_path).resolve())
    total_iteration = 0
    if curriculum.internal_eval_history:
        total_iteration = int(curriculum.internal_eval_history[-1]["iteration"])

    run_environment = os.environ.copy()
    for stage_index in range(int(args.max_stages)):
        current_distance = curriculum.current_max_distance
        next_distance = curriculum.next_distance
        target_iteration = total_iteration + int(args.iterations_per_stage)
        print(
            "V1 isolated stage {}: train {} iterations at {:.1f}m, eval {:.1f}+{:.1f}m".format(
                stage_index + 1,
                args.iterations_per_stage,
                current_distance,
                current_distance,
                next_distance,
            ),
            flush=True,
        )
        train_output = _run(
            build_training_command(
                sys.executable,
                repo_root,
                checkpoint,
                parent_checkpoint,
                state_path,
                args.iterations_per_stage,
                args.num_envs,
                framework_args,
            ),
            repo_root,
            run_environment,
        )
        log_directory = Path(parse_log_directory(train_output))
        checkpoint = str(log_directory / "model_{}.pt".format(args.iterations_per_stage))
        if not Path(checkpoint).is_file():
            raise FileNotFoundError("training checkpoint was not created: {}".format(checkpoint))

        eval_root = output_root / "iteration_{:04d}".format(target_iteration)
        eval_output = _run(
            build_evaluation_command(
                sys.executable,
                repo_root,
                checkpoint,
                current_distance,
                next_distance,
                args.episodes,
                args.eval_seed + stage_index,
                eval_root,
                framework_args,
                args.eval_num_envs,
            ),
            repo_root,
            run_environment,
        )
        del eval_output
        summary = json.loads((eval_root / "summary.json").read_text())
        result, gate, current, following = _record_external_evaluation(
            curriculum, target_iteration, summary
        )
        row = curriculum_history_row(
            target_iteration,
            result["level"],
            current_distance,
            next_distance,
            current,
            following,
            gate,
        )
        row["curriculum_pass"] = bool(result["pass"])
        row["promoted"] = bool(result["promoted"])
        _write_history_row(output_root / "curriculum_history.csv", row)
        _write_json(state_path, curriculum.to_dict())
        _write_json(
            output_root / "latest_stage.json",
            {
                "checkpoint": checkpoint,
                "iteration": target_iteration,
                "evaluation": summary,
                "gate": gate,
                "curriculum_result": result,
            },
        )
        total_iteration = target_iteration
        print(
            "V1 stage result: gate={} curriculum_pass={} promoted={} level={} max_distance={:.1f}m".format(
                gate["pass"],
                result["pass"],
                result["promoted"],
                curriculum.current_level,
                curriculum.current_max_distance,
            ),
            flush=True,
        )
        if curriculum.current_level == len(V1_CURRICULUM_LEVELS) - 1:
            break
    return {
        "checkpoint": checkpoint,
        "curriculum": curriculum.to_dict(),
        "output_dir": str(output_root),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_path", required=True)
    parser.add_argument("--parent_checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=os.getcwd())
    parser.add_argument("--iterations-per-stage", type=int, default=50)
    parser.add_argument("--max-stages", type=int, default=30)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--eval-num-envs", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parsed, framework_args = parser.parse_known_args(argv)
    run_curriculum(parsed, framework_args)


if __name__ == "__main__":
    main()
