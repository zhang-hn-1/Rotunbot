#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/lzq/workspace/SphericalRobot_LeggedGym-master-new-map"
PYTHON="/data/lzq/conda_envs/isaacgym/bin/python"
RUN_DIR="${1:?missing source run directory}"
CHECKPOINT="${2:?missing checkpoint}"

cd "$ROOT"
RUN_DIR="$(realpath "$RUN_DIR")"
OUTPUT_ROOT="$RUN_DIR/final_v49_validation_slope027/checkpoint_$CHECKPOINT"
mkdir -p "$OUTPUT_ROOT"

run_scan() {
    local output_dir="$OUTPUT_ROOT/reachable_domain"
    mkdir -p "$output_dir"
    CUDA_VISIBLE_DEVICES=2 "$PYTHON" \
        legged_gym/scripts/scan_vel_reachable_domain.py \
        --task rotunbot_vel_sru50_v49 \
        --headless \
        --sim_device cuda:0 \
        --rl_device cuda:0 \
        --load_run "$RUN_DIR" \
        --checkpoint "$CHECKPOINT" \
        --scan_direct_command_contract \
        --scan_settle_steps 250 \
        --scan_measure_steps 250 \
        --scan_output_dir "$output_dir" \
        >"$output_dir/scan.log" 2>&1
}

run_release() {
    local physical_gpu="$1"
    local seed="$2"
    local output_dir="$OUTPUT_ROOT/release_seed_$seed"
    mkdir -p "$output_dir"
    CUDA_VISIBLE_DEVICES="$physical_gpu" "$PYTHON" \
        legged_gym/scripts/evaluate_vel_tracking_release.py \
        --task rotunbot_vel_sru50_v49 \
        --headless \
        --sim_device cuda:0 \
        --rl_device cuda:0 \
        --load_run "$RUN_DIR" \
        --checkpoint "$CHECKPOINT" \
        --release_seed "$seed" \
        --release_output_dir "$output_dir" \
        >"$output_dir/evaluation.log" 2>&1
}

# Long constant grid and the first release seed run in parallel.
run_scan & pid2=$!
run_release 3 20260827 & pid3=$!
wait "$pid2"
wait "$pid3"

# Two additional held-out random sequences test release repeatability.
run_release 2 20260828 & pid2=$!
run_release 3 20260829 & pid3=$!
wait "$pid2"
wait "$pid3"

"$PYTHON" legged_gym/scripts/check_vel_release_gate.py \
    "$OUTPUT_ROOT/reachable_domain/reachable_domain_summary.json" \
    "$OUTPUT_ROOT"/release_seed_*/release_summary.json

tail -n 6 "$OUTPUT_ROOT/reachable_domain/scan.log"
for seed in 20260827 20260828 20260829; do
    tail -n 4 "$OUTPUT_ROOT/release_seed_$seed/evaluation.log"
done
echo "V49 final validation complete."
