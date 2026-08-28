"""Fail the process unless the reachable grid and all release suites pass."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reachable_summary")
    parser.add_argument("release_summaries", nargs="+")
    args = parser.parse_args()

    failures = []
    with open(args.reachable_summary, "r", encoding="utf-8") as handle:
        reachable = json.load(handle)
    if reachable.get("verdict") != "PASS":
        failures.append(
            "%s: %s (%s/%s points)"
            % (
                args.reachable_summary,
                reachable.get("verdict"),
                reachable.get("stable_reachable_points"),
                reachable.get("points"),
            )
        )

    for path in args.release_summaries:
        with open(path, "r", encoding="utf-8") as handle:
            release = json.load(handle)
        if release.get("verdict") != "PASS":
            failed_checks = [
                key for key, value in release.get("checks", {}).items()
                if not value
            ]
            failures.append(
                "%s: %s failed=%s"
                % (path, release.get("verdict"), ",".join(failed_checks))
            )

    if failures:
        raise SystemExit("velocity release gate failed:\n" + "\n".join(failures))
    print(
        "velocity release gate: PASS (%d release seeds)"
        % len(args.release_summaries)
    )


if __name__ == "__main__":
    main()
