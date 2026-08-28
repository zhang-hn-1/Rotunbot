"""Combine Experiment A and B into the root-cause decision matrix."""

import argparse
import json
from pathlib import Path

def summarize(action_report_path, policy_report_path, output_path):
    action_report = json.loads(Path(action_report_path).read_text(encoding="utf-8"))
    policy_report = json.loads(Path(policy_report_path).read_text(encoding="utf-8"))
    action_decision = action_report["decision"]
    metrics = policy_report["metrics"]
    b_weak = metrics["a1_response_span"] < 0.10 or abs(metrics["spearman_gy_a1"]) < 0.5
    b_strong = metrics["a1_response_span"] >= 0.25 and abs(metrics["spearman_gy_a1"]) >= 0.5
    b_label = "B-WEAK" if b_weak else ("B-STRONG-SENSITIVITY" if b_strong else "B-MIXED")
    if action_decision == "A-FAIL":
        conclusion = "低层 action/controller 问题：Experiment A 表明 action1 不能提供可靠横向控制。"
    elif action_decision == "A-WEAK" and b_weak:
        conclusion = "控制偏弱 + PPO shortcut：action1能力不足且策略没有稳定利用 gy。"
    elif action_decision == "A-GOOD" and b_weak:
        conclusion = "RL/reward/采样设计问题：低层横移可用，但策略忽略 gy。"
    elif action_decision == "A-GOOD" and b_strong:
        conclusion = "动态闭环/刹车/credit assignment 问题：物理控制和静态 gy 响应都存在。"
    else:
        conclusion = "混合或未决：需结合动作轨迹和闭环 rollout 进一步定位。"
    report = {
        "experiment_a_decision": action_decision,
        "experiment_b_metrics": metrics,
        "b_weak": b_weak,
        "b_strong": b_strong,
        "b_label": b_label,
        "conclusion": conclusion,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "# Depth-local root-cause summary\n\n"
        f"- Experiment A: **{action_decision}**\n"
        f"- Experiment B: **{b_label}**\n"
        f"- Experiment B a1 response span: `{metrics['a1_response_span']:.6f}`\n"
        f"- Experiment B Pearson(gy, a1): `{metrics['pearson_gy_a1']:.6f}`\n"
        f"- Experiment B Spearman(gy, a1): `{metrics['spearman_gy_a1']:.6f}`\n"
        f"- Experiment B sign agreement: `{metrics['sign_agreement_rate']:.6f}`\n"
        f"- Experiment B symmetry error: `{metrics['symmetry_error']:.6f}`\n\n"
        f"## Conclusion\n\n{conclusion}\n\n"
        "No long PPO training was started in this diagnostic round.\n",
        encoding="utf-8",
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-report", default="logs/depth_local_diagnostics/action_mapping_sweep.json")
    parser.add_argument("--policy-report", default="logs/depth_local_diagnostics/policy_gy_sweep.json")
    parser.add_argument("--output", default="logs/depth_local_diagnostics/depth_local_root_cause_summary.md")
    args = parser.parse_args(argv)
    report = summarize(Path(args.action_report), Path(args.policy_report), Path(args.output))
    print(report["conclusion"])


if __name__ == "__main__":
    main()
