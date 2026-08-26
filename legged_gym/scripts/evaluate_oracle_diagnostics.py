"""Diagnostic entry point for the unchanged Oracle Raw/Reachability executor."""

def parse_script_args():
    from legged_gym.scripts.evaluate_oracle_maze import _parse_script_args

    return _parse_script_args()


def run_gate(args, script_args):
    from legged_gym.scripts.evaluate_oracle_maze import run_gate as oracle_run_gate

    return oracle_run_gate(args, script_args)


def main():
    from legged_gym.scripts.evaluate_oracle_maze import _isaac_args

    return run_gate(_isaac_args(), parse_script_args())


if __name__ == "__main__":
    main()
