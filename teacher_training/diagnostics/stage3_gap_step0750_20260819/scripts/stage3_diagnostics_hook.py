"""Runtime-only Stage-3 diagnostics for the official DoorMan environment.

This module installs an import hook so the official source tree stays clean.
It records metrics only and does not alter observations, rewards, actions,
termination conditions, stage transitions, or environment state.
"""

from __future__ import annotations

import csv
import importlib.abc
import importlib.util
import json
import os
import sys
from pathlib import Path


TARGET_MODULE = "gr00t.rl.envs.door.door_open_homie"


class Stage3Recorder:
    def __init__(self, env):
        import torch

        output_dir = Path(os.environ["DOORMAN_DIAG_DIR"])
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.step = 0
        self.flush_interval = int(os.environ.get("DOORMAN_DIAG_FLUSH_INTERVAL", "25"))
        self.policy_dt = float(getattr(env, "dt", 0.02))

        num_envs = int(env.num_envs)
        self.episode_id = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        self.first_episode_done = torch.zeros(
            num_envs, device=env.device, dtype=torch.bool
        )
        self.stage3_duration = torch.zeros(num_envs, device=env.device, dtype=torch.long)
        self.last_stage = env.stage_buf.detach().clone()
        self.last_episode_length = env.episode_length_buf.detach().clone()
        self.initialized = False

        self.stage3_enter_count = 0
        self.stage3_to_stage4_count = 0
        self.stage4_reach_count = 0
        self.stage3_rows = 0

        self.timeseries_handle = (output_dir / "stage3_timeseries.csv").open(
            "w", newline="", buffering=1
        )
        self.events_handle = (output_dir / "stage_events.csv").open(
            "w", newline="", buffering=1
        )
        self.timeseries = csv.writer(self.timeseries_handle)
        self.events = csv.writer(self.events_handle)
        self.timeseries.writerow(
            [
                "global_eval_step",
                "policy_time_s",
                "env_id",
                "episode_id",
                "stage3_duration_steps",
                "stage3_duration_s",
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
        )
        self.events.writerow(
            [
                "global_eval_step",
                "env_id",
                "episode_id",
                "event",
                "from_stage",
                "to_stage",
                "stage3_duration_steps",
            ]
        )

        metadata = {
            "schema_version": 1,
            "num_envs": num_envs,
            "policy_dt": self.policy_dt,
            "reward_modified": False,
            "environment_modified": False,
            "force_sensor_semantics": "raw object-to-hand contact force from official sensor",
            "force_on_door_semantics": "negative of object-to-hand force (Newton third-law estimate)",
            "hinge_origin": "official generated hinge local position transformed by door root quaternion",
            "application_point": "official grasp_target world position",
            "tau_hinge_sensor_z": "world-Z component of cross(grasp_target-hinge_origin, official_force_sensor)",
            "tau_open": "cross(grasp_target-hinge_origin, official_force_sensor) projected on positive generated hinge axis",
            "force_tangential_alignment": "tau_open/(norm(radial_lever)*norm(official_force_sensor))",
        }
        (output_dir / "diagnostic_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    def _write_event(self, env_ids, name, previous_stage, current_stage):
        if env_ids.numel() == 0:
            return
        ids = env_ids.detach().cpu().tolist()
        episodes = self.episode_id[env_ids].detach().cpu().tolist()
        durations = self.stage3_duration[env_ids].detach().cpu().tolist()
        previous = previous_stage[env_ids].detach().cpu().tolist()
        current = current_stage[env_ids].detach().cpu().tolist()
        self.events.writerows(
            [self.step, i, ep, name, old, new, duration]
            for i, ep, old, new, duration in zip(ids, episodes, previous, current, durations)
        )

    def capture(self, env, previous_stage, previous_episode_length):
        import torch

        current_stage = env.stage_buf.detach()
        current_episode_length = env.episode_length_buf.detach()
        # Evaluation keeps stepping environments that already finished while it
        # waits for the slowest environment. Detect the reset between callbacks
        # and retain only each environment's first episode.
        reset_mask = current_episode_length < self.last_episode_length
        newly_completed = reset_mask & ~self.first_episode_done
        self.first_episode_done[newly_completed] = True
        self.episode_id[reset_mask] += 1
        self.stage3_duration[reset_mask] = 0
        active = ~self.first_episode_done

        if not self.initialized:
            entered_stage3 = (current_stage == 3) & active
            self.initialized = True
        else:
            entered_stage3 = (current_stage == 3) & (
                (previous_stage != 3) | (self.last_stage != 3)
            ) & active
        stage3_to_stage4 = (
            (previous_stage == 3) & (current_stage >= 4) & ~reset_mask & active
        )
        reached_stage4 = (
            (previous_stage < 4) & (current_stage >= 4) & ~reset_mask & active
        )

        self.stage3_enter_count += int(entered_stage3.sum().item())
        self.stage3_to_stage4_count += int(stage3_to_stage4.sum().item())
        self.stage4_reach_count += int(reached_stage4.sum().item())
        self._write_event(
            torch.where(entered_stage3)[0], "stage3_enter", previous_stage, current_stage
        )
        self._write_event(
            torch.where(stage3_to_stage4)[0],
            "stage3_to_stage4",
            previous_stage,
            current_stage,
        )
        extra_stage4 = reached_stage4 & ~stage3_to_stage4
        self._write_event(
            torch.where(extra_stage4)[0], "stage4_reach", previous_stage, current_stage
        )

        in_stage3 = (current_stage == 3) & active
        self.stage3_duration[in_stage3] += 1
        self.stage3_duration[~in_stage3] = 0
        env_ids = torch.where(in_stage3)[0]

        if env_ids.numel() > 0:
            left_forces = env.simulator.object_to_hand_contact_forces[
                :, 0, env.left_hand_indices_tgt_ct_sensor, :
            ]
            right_forces = env.simulator.object_to_hand_contact_forces[
                :, 0, env.right_hand_indices_tgt_ct_sensor, :
            ]
            left_net = left_forces.sum(dim=-2)
            right_net = right_forces.sum(dim=-2)
            use_left = env.door_open_lr < 0
            force_sensor = torch.where(use_left[:, None], left_net, right_net)
            force_on_door = -force_sensor

            left_contacts = (left_forces.norm(dim=-1) > 1.0).sum(dim=-1)
            right_contacts = (right_forces.norm(dim=-1) > 1.0).sum(dim=-1)
            contact_count = torch.where(use_left, left_contacts, right_contacts)

            door = env.simulator.scene.articulations["door"]
            hinge_angle = door.data.joint_pos[:, 0]
            hinge_velocity = door.data.joint_vel[:, 0]
            handle_angle = door.data.joint_pos[:, 1]
            application_point = env._compute_grasp_target()

            # Match the generated joint in env_rand/door.py. Its local
            # position is (0.02, -0.5*width*door_open_lr, 0); its positive
            # axis is +Z for right-hand doors and -Z for left-hand doors.
            root_quat = door.data.root_quat_w  # wxyz
            hinge_local = torch.stack(
                [
                    torch.full_like(env.door_width, 0.02),
                    -0.5 * env.door_width * env.door_open_lr,
                    torch.zeros_like(env.door_width),
                ],
                dim=-1,
            )
            axis_local = torch.stack(
                [
                    torch.zeros_like(env.door_width),
                    torch.zeros_like(env.door_width),
                    -env.door_open_lr,
                ],
                dim=-1,
            )

            def quat_apply_wxyz(quat, vector):
                xyz = quat[:, 1:]
                uv = torch.cross(xyz, vector, dim=-1)
                uuv = torch.cross(xyz, uv, dim=-1)
                return vector + 2.0 * (quat[:, :1] * uv + uuv)

            hinge_origin = door.data.root_pos_w + quat_apply_wxyz(root_quat, hinge_local)
            hinge_axis = quat_apply_wxyz(root_quat, axis_local)
            lever = application_point - hinge_origin
            torque_vector = torch.cross(lever, force_sensor, dim=-1)
            tau_hinge_sensor = torque_vector[:, 2]
            tau_open = (torque_vector * hinge_axis).sum(dim=-1)
            radial_lever = lever - (lever * hinge_axis).sum(dim=-1, keepdim=True) * hinge_axis
            denominator = radial_lever.norm(dim=-1) * force_sensor.norm(dim=-1)
            alignment = (tau_open / denominator.clamp_min(1.0e-6)).clamp(-1.0, 1.0)

            packed = torch.cat(
                [
                    env_ids[:, None].float(),
                    self.episode_id[env_ids, None].float(),
                    self.stage3_duration[env_ids, None].float(),
                    hinge_angle[env_ids, None],
                    hinge_velocity[env_ids, None],
                    handle_angle[env_ids, None],
                    force_sensor[env_ids],
                    force_on_door[env_ids],
                    tau_hinge_sensor[env_ids, None],
                    tau_open[env_ids, None],
                    alignment[env_ids, None],
                    contact_count[env_ids, None].float(),
                ],
                dim=1,
            ).detach().cpu().tolist()

            rows = []
            for row in packed:
                env_id = int(row[0])
                episode_id = int(row[1])
                duration_steps = int(row[2])
                rows.append(
                    [
                        self.step,
                        self.step * self.policy_dt,
                        env_id,
                        episode_id,
                        duration_steps,
                        duration_steps * self.policy_dt,
                        *row[3:],
                    ]
                )
            self.timeseries.writerows(rows)
            self.stage3_rows += len(rows)

        if self.step % self.flush_interval == 0:
            self.timeseries_handle.flush()
            self.events_handle.flush()
            self.write_live_summary()

        self.last_stage = current_stage.clone()
        self.last_episode_length = current_episode_length.clone()
        self.step += 1

    def write_live_summary(self):
        denominator = max(self.stage3_enter_count, 1)
        payload = {
            "global_eval_step": self.step,
            "stage3_enter_count": self.stage3_enter_count,
            "stage3_to_stage4_count": self.stage3_to_stage4_count,
            "stage4_reach_count": self.stage4_reach_count,
            "stage4_reach_rate_per_stage3_entry": self.stage3_to_stage4_count / denominator,
            "stage3_timeseries_rows": self.stage3_rows,
        }
        temporary = self.output_dir / "live_summary.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.output_dir / "live_summary.json")


def _patch_module(module):
    cls = module.DoorPregrasp
    if getattr(cls, "_stage3_diagnostics_installed", False):
        return

    original = cls._post_compute_observations_callback

    def wrapped(self, *args, **kwargs):
        previous_stage = self.stage_buf.detach().clone()
        previous_episode_length = self.episode_length_buf.detach().clone()
        result = original(self, *args, **kwargs)
        if not getattr(self, "is_evaluating", False):
            return result
        recorder = getattr(self, "_stage3_diagnostics_recorder", None)
        if recorder is None:
            recorder = Stage3Recorder(self)
            self._stage3_diagnostics_recorder = recorder
        recorder.capture(self, previous_stage, previous_episode_length)
        return result

    cls._post_compute_observations_callback = wrapped
    cls._stage3_diagnostics_installed = True
    print("[stage3-diagnostics] instrumentation installed; rewards unchanged")


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, original_loader):
        self.original_loader = original_loader

    def create_module(self, spec):
        create = getattr(self.original_loader, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self.original_loader.exec_module(module)
        _patch_module(module)


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET_MODULE:
            return None
        try:
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not locate {fullname}")
        spec.loader = _PatchLoader(spec.loader)
        return spec


def install_import_hook():
    if TARGET_MODULE in sys.modules:
        _patch_module(sys.modules[TARGET_MODULE])
        return
    sys.meta_path.insert(0, _PatchFinder())
