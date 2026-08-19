"""Summarize Stage-3 diagnostics and plot the longest Stage-3 rollout."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostic_dir", type=Path)
    args = parser.parse_args()
    directory = args.diagnostic_dir

    with (directory / "stage3_timeseries.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("No Stage-3 rows were recorded")

    numeric = {}
    numeric_names = [
        "hinge_angle",
        "hinge_velocity",
        "handle_angle",
        "force_sensor_x",
        "force_sensor_y",
        "force_sensor_z",
        "force_on_door_x",
        "force_on_door_y",
        "force_on_door_z",
        "tau_hinge_sensor_z",
        "tau_open",
        "force_tangential_alignment",
        "handle_contact_count",
    ]
    for name in numeric_names:
        numeric[name] = np.asarray([float(row[name]) for row in rows], dtype=np.float64)

    with (directory / "stage_events.csv").open(newline="") as handle:
        events = list(csv.DictReader(handle))
    enter_count = sum(event["event"] == "stage3_enter" for event in events)
    transition_count = sum(event["event"] == "stage3_to_stage4" for event in events)

    segments = defaultdict(list)
    for row in rows:
        segments[(int(row["env_id"]), int(row["episode_id"]))].append(row)
    longest_key, longest = max(segments.items(), key=lambda item: len(item[1]))
    episode_durations = np.asarray(
        [max(int(row["stage3_duration_steps"]) for row in segment) for segment in segments.values()],
        dtype=np.float64,
    )

    positive_velocity = numeric["hinge_velocity"][numeric["hinge_velocity"] > 0]
    summary = {
        "stage3_enter_count": enter_count,
        "stage3_to_stage4_count": transition_count,
        "stage4_reach_rate_per_stage3_entry": transition_count / max(enter_count, 1),
        "stage3_rows": len(rows),
        "stage3_rollout_segments": len(segments),
        "hinge_angle_mean": float(numeric["hinge_angle"].mean()),
        "hinge_angle_p95": percentile(numeric["hinge_angle"], 95),
        "hinge_angle_max": float(numeric["hinge_angle"].max()),
        "hinge_velocity_mean": float(numeric["hinge_velocity"].mean()),
        "hinge_velocity_positive_p95": percentile(positive_velocity, 95),
        "handle_angle_mean": float(numeric["handle_angle"].mean()),
        "handle_angle_p95": percentile(numeric["handle_angle"], 95),
        "raw_hand_force_xyz_mean": [
            float(numeric[name].mean())
            for name in ("force_sensor_x", "force_sensor_y", "force_sensor_z")
        ],
        "raw_hand_force_xyz_abs_p95": [
            percentile(np.abs(numeric[name]), 95)
            for name in ("force_sensor_x", "force_sensor_y", "force_sensor_z")
        ],
        "tau_hinge_sensor_mean": float(numeric["tau_hinge_sensor_z"].mean()),
        "tau_open_mean": float(numeric["tau_open"].mean()),
        "tau_open_abs_p95": percentile(np.abs(numeric["tau_open"]), 95),
        "tau_open_p95": percentile(numeric["tau_open"], 95),
        "tau_open_max": float(numeric["tau_open"].max()),
        "force_tangential_alignment_mean": float(
            numeric["force_tangential_alignment"].mean()
        ),
        "force_tangential_alignment_p95": percentile(
            numeric["force_tangential_alignment"], 95
        ),
        "force_tangential_alignment_positive_fraction": float(
            (numeric["force_tangential_alignment"] > 0).mean()
        ),
        "stage3_duration_steps_mean": float(episode_durations.mean()),
        "stage3_duration_steps_p95": percentile(episode_durations, 95),
        "stage3_duration_steps_max": int(episode_durations.max()),
        "handle_contact_count_mean": float(numeric["handle_contact_count"].mean()),
        "handle_contact_count_p95": percentile(numeric["handle_contact_count"], 95),
        "valid_handle_contact_fraction": float(
            (numeric["handle_contact_count"] >= 4).mean()
        ),
        "longest_rollout": {
            "env_id": longest_key[0],
            "episode_id": longest_key[1],
            "steps": len(longest),
        },
    }

    correlations = {}
    for x_name in ("force_sensor_x", "tau_open"):
        for y_name in ("hinge_velocity", "hinge_angle"):
            x = numeric[x_name]
            y = numeric[y_name]
            correlations[f"{x_name}_vs_{y_name}"] = (
                float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None
            )
    summary["correlations"] = correlations
    (directory / "stage3_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    longest.sort(key=lambda row: int(row["stage3_duration_steps"]))
    time_s = np.asarray([float(row["stage3_duration_s"]) for row in longest])
    figure, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(time_s, [float(row["force_sensor_x"]) for row in longest], label="Fx sensor")
    axes[0].plot(time_s, [float(row["force_on_door_x"]) for row in longest], label="Fx on door", alpha=0.8)
    axes[0].set_ylabel("Force (N)")
    axes[0].legend(loc="best")

    axes[1].plot(time_s, [float(row["tau_open"]) for row in longest], color="tab:orange")
    axes[1].set_ylabel("tau_open (N m)")

    axes[2].plot(
        time_s,
        [float(row["hinge_velocity"]) for row in longest],
        color="tab:green",
    )
    axes[2].set_ylabel("Hinge vel (rad/s)")

    axes[3].plot(
        time_s,
        [float(row["hinge_angle"]) for row in longest],
        color="tab:red",
    )
    axes[3].axhline(np.deg2rad(10.0), color="black", linestyle="--", label="Stage 4 threshold")
    axes[3].set_ylabel("Hinge angle (rad)")
    axes[3].set_xlabel("Time within Stage 3 (s)")
    axes[3].legend(loc="best")
    figure.suptitle(
        f"DoorMan Stage-3 longest rollout: env {longest_key[0]}, episode {longest_key[1]}"
    )
    figure.tight_layout()
    figure.savefig(directory / "stage3_timeline_longest.png", dpi=160)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
