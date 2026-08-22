#!/usr/bin/env python3
"""Compare two schema-v2 DoorMan post-open diagnostic results.

The report deliberately uses reward-independent physical/action metrics. It is
intended for fixed-checkpoint gates, not for selecting a policy by shaped return.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARM_METRICS = (
    "left_arm_rest_error_normalized",
    "right_arm_rest_error_normalized",
    "left_proximal_rest_error_normalized",
    "right_proximal_rest_error_normalized",
    "left_wrist_rest_error_normalized",
    "right_wrist_rest_error_normalized",
    "left_arm_action_delta_rms",
    "right_arm_action_delta_rms",
)

ERROR_METRICS = ARM_METRICS[:6]
SLOPE_METRICS = ERROR_METRICS


def load_analysis(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 2:
        raise ValueError(f"expected schema_version=2: {path}")
    if int(data.get("episode_count", 0)) <= 0:
        raise ValueError(f"missing completed episodes: {path}")
    return data


def optional_float(raw: Any) -> float | None:
    return None if raw is None else float(raw)


def value(data: dict[str, Any], quartile: str, metric: str) -> float | None:
    return optional_float(
        data["quartiles_by_normalized_post_open_progress"][quartile][metric]["mean"]
    )


def slope(data: dict[str, Any], metric: str) -> float | None:
    return optional_float(
        data["per_episode_linear_slopes_per_second"][metric]["mean"]
    )


def finite_ratio(
    numerator: float | None, denominator: float | None
) -> float | None:
    if numerator is None or denominator is None or abs(denominator) <= 1.0e-12:
        return None
    return numerator / denominator


def difference(candidate: float | None, anchor: float | None) -> float | None:
    if candidate is None or anchor is None:
        return None
    return candidate - anchor


def less_than(value_: float | None, threshold: float) -> bool:
    return value_ is not None and value_ < threshold


def no_more_than(value_: float | None, threshold: float) -> bool:
    return value_ is not None and value_ <= threshold


def comparison(anchor: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    anchor_success = float(anchor["success_rate"])
    candidate_success = float(candidate["success_rate"])
    metrics: dict[str, Any] = {}
    for metric in ARM_METRICS:
        anchor_q2 = value(anchor, "Q2", metric)
        anchor_q4 = value(anchor, "Q4", metric)
        candidate_q2 = value(candidate, "Q2", metric)
        candidate_q4 = value(candidate, "Q4", metric)
        item: dict[str, Any] = {
            "anchor_q2": anchor_q2,
            "anchor_q4": anchor_q4,
            "candidate_q2": candidate_q2,
            "candidate_q4": candidate_q4,
            "candidate_minus_anchor_q4": difference(candidate_q4, anchor_q4),
            "anchor_q4_over_q2": finite_ratio(anchor_q4, anchor_q2),
            "candidate_q4_over_q2": finite_ratio(candidate_q4, candidate_q2),
        }
        if metric in SLOPE_METRICS:
            anchor_slope = slope(anchor, metric)
            candidate_slope = slope(candidate, metric)
            item.update(
                {
                    "anchor_slope_per_s": anchor_slope,
                    "candidate_slope_per_s": candidate_slope,
                    "candidate_minus_anchor_slope_per_s": difference(
                        candidate_slope, anchor_slope
                    ),
                }
            )
        metrics[metric] = item

    gates = {
        "success_drop_le_5pp": candidate_success >= anchor_success - 0.05,
        "left_arm_q4_over_q2_le_1_2": (
            metrics["left_arm_rest_error_normalized"]["candidate_q4_over_q2"]
            is not None
            and no_more_than(
                metrics["left_arm_rest_error_normalized"][
                    "candidate_q4_over_q2"
                ],
                1.2,
            )
        ),
        "right_arm_q4_over_q2_le_1_2": (
            metrics["right_arm_rest_error_normalized"]["candidate_q4_over_q2"]
            is not None
            and no_more_than(
                metrics["right_arm_rest_error_normalized"][
                    "candidate_q4_over_q2"
                ],
                1.2,
            )
        ),
        "all_error_slopes_le_0_005_per_s": all(
            no_more_than(metrics[name]["candidate_slope_per_s"], 0.005)
            for name in SLOPE_METRICS
        ),
        "left_wrist_q4_lt_0_12": less_than(
            metrics["left_wrist_rest_error_normalized"]["candidate_q4"], 0.12
        ),
        "right_proximal_q4_lt_0_06": less_than(
            metrics["right_proximal_rest_error_normalized"]["candidate_q4"],
            0.06,
        ),
        "both_arm_action_q4_lt_0_25": all(
            less_than(metrics[name]["candidate_q4"], 0.25)
            for name in (
                "left_arm_action_delta_rms",
                "right_arm_action_delta_rms",
            )
        ),
    }

    direction_metrics = ERROR_METRICS + ARM_METRICS[6:]
    comparable_q4 = [
        name
        for name in direction_metrics
        if metrics[name]["candidate_q4"] is not None
        and metrics[name]["anchor_q4"] is not None
    ]
    q4_improved = sum(
        metrics[name]["candidate_q4"] < metrics[name]["anchor_q4"]
        for name in comparable_q4
    )
    comparable_slopes = [
        name
        for name in SLOPE_METRICS
        if metrics[name]["candidate_slope_per_s"] is not None
        and metrics[name]["anchor_slope_per_s"] is not None
    ]
    slope_improved = sum(
        metrics[name]["candidate_slope_per_s"]
        < metrics[name]["anchor_slope_per_s"]
        for name in comparable_slopes
    )

    return {
        "anchor": {
            "episodes": int(anchor["episode_count"]),
            "successful_episodes": int(anchor["successful_episode_count"]),
            "success_rate": anchor_success,
        },
        "candidate": {
            "episodes": int(candidate["episode_count"]),
            "successful_episodes": int(candidate["successful_episode_count"]),
            "success_rate": candidate_success,
        },
        "success_rate_delta": candidate_success - anchor_success,
        "physical_comparison_available": bool(comparable_q4),
        "metrics": metrics,
        "direction_summary": {
            "q4_metrics_improved": q4_improved,
            "q4_metrics_compared": len(comparable_q4),
            "error_slopes_improved": slope_improved,
            "error_slopes_compared": len(comparable_slopes),
        },
        "acceptance_gates": gates,
        "all_final_acceptance_gates_pass": all(gates.values()),
    }


def fmt(value_: float | None) -> str:
    return "n/a" if value_ is None else f"{value_:.4f}"


def fmt_signed(value_: float | None) -> str:
    return "n/a" if value_ is None else f"{value_:+.4f}"


def markdown(report: dict[str, Any]) -> str:
    anchor = report["anchor"]
    candidate = report["candidate"]
    lines = [
        "# Arm reward-v2 paired diagnostic",
        "",
        (
            f"Anchor success: {anchor['successful_episodes']}/{anchor['episodes']} "
            f"({anchor['success_rate']:.2%}); candidate success: "
            f"{candidate['successful_episodes']}/{candidate['episodes']} "
            f"({candidate['success_rate']:.2%}); delta "
            f"{report['success_rate_delta']:+.2%}."
        ),
        "",
        "| Metric | Anchor Q4 | Candidate Q4 | Delta | Candidate Q4/Q2 | Anchor slope/s | Candidate slope/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ARM_METRICS:
        item = report["metrics"][name]
        lines.append(
            "| "
            + " | ".join(
                (
                    name,
                    fmt(item["anchor_q4"]),
                    fmt(item["candidate_q4"]),
                    fmt_signed(item["candidate_minus_anchor_q4"]),
                    fmt(item["candidate_q4_over_q2"]),
                    fmt(item.get("anchor_slope_per_s")),
                    fmt(item.get("candidate_slope_per_s")),
                )
            )
            + " |"
        )

    direction = report["direction_summary"]
    if not report["physical_comparison_available"]:
        lines.extend(
            [
                "",
                (
                    "The candidate produced no comparable successful post-open "
                    "trajectory quartiles. Physical entries are `n/a`; this is "
                    "missing successful behavior, not a passing metric."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Direction summary",
            "",
            (
                f"Q4 improved on {direction['q4_metrics_improved']}/"
                f"{direction['q4_metrics_compared']} arm/action metrics; error "
                f"slope improved on {direction['error_slopes_improved']}/"
                f"{direction['error_slopes_compared']} metrics."
            ),
            "",
            "## Predeclared final acceptance gates",
            "",
        ]
    )
    for name, passed in report["acceptance_gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(
        [
            "",
            (
                "This early gate reports direction and safety. Failure to meet all "
                "final targets at one checkpoint does not by itself authorize changing the "
                "frozen training scheme."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("anchor", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()

    report = comparison(load_analysis(args.anchor), load_analysis(args.candidate))
    rendered = markdown(report)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.write_text(rendered, encoding="utf-8")
    if not args.json_out and not args.markdown_out:
        print(rendered, end="")


if __name__ == "__main__":
    main()
