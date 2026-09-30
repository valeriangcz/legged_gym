"""Parallel mass-sweep recorder for the HexClimb expert.

Edit the constants in the "Manual experiment configuration" block, then run:

    python legged_gym/scripts/record_expert_climb.py --task hex_climb --headless

All outputs are written below ``logs/expert_climb``.  The companion README in
each run directory is the authoritative field dictionary for the HDF5 file.
"""

from __future__ import annotations

import copy
import json
import math
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Sequence, Tuple

# Isaac Gym must initialize before PyTorch.  Import it explicitly here rather
# than relying on a later legged_gym import to do so indirectly.
from isaacgym import gymapi  # noqa: F401

import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import HexClimbCfg
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict


# ---------------------------------------------------------------------------
# Manual experiment configuration
# ---------------------------------------------------------------------------
# World gravity is R_x(angle) @ [0, 0, -GRAVITY_MAGNITUDE].  A positive 90 deg
# angle therefore produces gravity along +Y in the world frame.
GRAVITY_X_ANGLES_DEG: Tuple[float, ...] = (0.0,60.0 ,90.0 ,120.0,180.0)

GRAVITY_MAGNITUDE = 9.81  # m/s^2; set to 0.0 for the zero-gravity test.

# One actor/environment is created for every item in this tuple.  They run in
# parallel within each gravity/adsorption case.
# 0.05,
MASS_SCALES: Tuple[float, ...] = (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,)
SUCTION_FORCE_MAXES: Tuple[float, ...] = (150.0,)  # N per cup
SUCTION_FORCE_DELTAS: Tuple[float, ...] = (150.0,)  # N per control updat3

FIXED_STATIC_FRICTION = 0.7
FIXED_DYNAMIC_FRICTION = 0.7
FIXED_ROBOT_SHAPE_FRICTION = 0.7

TRANSITION_DURATION_S = 0.5
RECORD_DURATION_S = 5.0
SETTLE_DURATION_S = 0.5
H5_CHUNK_FRAMES = 400  # one second at the default 400 Hz physical-step rate
RUN_LABEL = "mass_sweep"
DATASET_VERSION = "expert_climb_h5_v2"
README_TEMPLATE_PATH = Path(__file__).with_name("record_expert_climb_README.md")


FIELD_DESCRIPTIONS: Dict[str, Tuple[str, str]] = {
    "joint_position": ("rad", "All URDF DOF positions; DOF order is stored in the group metadata."),
    "joint_velocity": ("rad/s", "All URDF DOF velocities in the same order as joint_position."),
    "actuator_torque": ("N m", "Torque actually submitted to PhysX after controller calculation and clipping."),
    "feedforward_torque": ("N m", "Expert main-motor feedforward torque before it is added to position-control torque and before total actuator clipping; order is motor_dof_names."),
    "dof_generalized_force": ("N m", "Isaac Gym DOF force-sensor reading; includes generalized constraint effects."),
    "joint_sensor_wrench_world_raw": ("N, N m", "Raw force-sensor wrench in world axes at the child-link sensor origin."),
    "joint_sensor_origin_world": ("m", "World position of every joint force-sensor origin."),
    "joint_parent_pose_world": ("m, quaternion xyzw", "Parent link pose used to transform each joint wrench."),
    "joint_wrench_parent": ("N, N m", "[F,M] at the parent-link origin in current parent axes; positive is downstream assembly acting on parent."),
    "com_velocity_world": ("m/s", "Mass-weighted true whole-robot COM velocity in world axes."),
    "com_velocity_body": ("m/s", "True whole-robot COM velocity expressed in the root/body axes."),
    "desired_velocity_body": ("m/s, m/s, rad/s", "Expert command [vx, vy, yaw_rate] in body axes."),
    "adhesion_command_wrench_world": ("N, N m", "[F,M] commanded by the three applied adhesion forces about the toe/cup reference point."),
    "surface_force_world": ("N", "Net environment force on suck/toe/empty bodies. Its negative is robot force on the surface."),
    "cup_interface_wrench_parent": ("N, N m", "foot -> ball1 interface wrench, at foot origin in current foot axes; positive cup assembly load on foot."),
    "cup_reference_pose_world": ("m, quaternion xyzw", "Toe-body pose used as each cup reference frame."),
    "time_s": ("s", "Elapsed recorded physical simulation time within this case; transition and settle frames are excluded."),
    "segment_id": ("index", "Index into the group-level segment_names and segment_target_command datasets."),
    "sample_valid": ("bool", "Always one for recorded stable samples; present to make filtering explicit."),
}


def _as_jsonable(value: Any) -> Any:
    """Convert nested config values into JSON/YAML-safe primitives."""
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=LEGGED_GYM_ROOT_DIR,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _make_run_dir() -> Path:
    root = Path(LEGGED_GYM_ROOT_DIR) / "logs" / "expert_climb"
    root.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{RUN_LABEL}"
    candidate = root / base
    suffix = 1
    while candidate.exists():
        candidate = root / f"{base}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _gravity_from_x_angle(angle_deg: float) -> List[float]:
    angle = math.radians(angle_deg)
    return [
        0.0,
        GRAVITY_MAGNITUDE * math.sin(angle),
        -GRAVITY_MAGNITUDE * math.cos(angle),
    ]


def _command_segments(cfg: HexClimbCfg) -> List[Tuple[str, Tuple[float, float, float]]]:
    x_min, x_max = map(float, cfg.commands.ranges.lin_vel_x)
    y_min, y_max = map(float, cfg.commands.ranges.lin_vel_y)
    yaw_min, yaw_max = map(float, cfg.commands.ranges.ang_vel_yaw)
    return [
        ("+y", (0.0, y_max, 0.0)),
        ("-y", (0.0, y_min, 0.0)),
        ("+x", (x_max, 0.0, 0.0)),
        ("-x", (x_min, 0.0, 0.0)),
        ("+x+y", (x_max, y_max, 0.0)),
        ("+x-y", (x_max, y_min, 0.0)),
        ("-x+y", (x_min, y_max, 0.0)),
        ("-x-y", (x_min, y_min, 0.0)),
        ("+y+yaw", (0.0, y_max, yaw_max)),
        ("+y-yaw", (0.0, y_max, yaw_min)),
        ("-y+yaw", (0.0, y_min, yaw_max)),
        ("-y-yaw", (0.0, y_min, yaw_min)),
    ]


class H5CaseWriter:
    """Append synchronized [time, mass_scale, ...] samples without RAM growth."""

    def __init__(self, h5_file, group, mass_count: int, chunk_frames: int):
        self.h5_file = h5_file
        self.group = group
        self.data_group = group.create_group("data")
        self.mass_count = mass_count
        self.chunk_frames = chunk_frames
        self.datasets = {}
        self.buffer: Dict[str, List[np.ndarray]] = {}

    def write_static(self, name: str, data: Any, **attrs: Any) -> None:
        dataset = self.group.create_dataset(name, data=data)
        for key, value in attrs.items():
            dataset.attrs[key] = value

    def append(self, telemetry: Dict[str, torch.Tensor], time_s: float, segment_id: int) -> None:
        sample: Dict[str, np.ndarray] = {
            key: value.detach().cpu().numpy().astype(np.float32, copy=False)
            for key, value in telemetry.items()
        }
        for key, value in sample.items():
            if value.shape[0] != self.mass_count:
                raise RuntimeError(
                    f"{key} has mass dimension {value.shape[0]}, expected {self.mass_count}"
                )
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise FloatingPointError(f"non-finite value in recording field: {key}")
            self.buffer.setdefault(key, []).append(value)
        self.buffer.setdefault("time_s", []).append(np.asarray(time_s, dtype=np.float64))
        self.buffer.setdefault("segment_id", []).append(np.asarray(segment_id, dtype=np.int16))
        self.buffer.setdefault("sample_valid", []).append(np.asarray(1, dtype=np.uint8))
        if len(self.buffer["time_s"]) >= self.chunk_frames:
            self.flush()

    def _dataset_for(self, name: str, sample: np.ndarray):
        if name in self.datasets:
            return self.datasets[name]
        tail = sample.shape
        chunks = (self.chunk_frames,) + tail
        dataset = self.data_group.create_dataset(
            name,
            shape=(0,) + tail,
            maxshape=(None,) + tail,
            chunks=chunks,
            compression="lzf",
            dtype=sample.dtype,
        )
        unit, description = FIELD_DESCRIPTIONS.get(name, ("", ""))
        dataset.attrs["unit"] = unit
        dataset.attrs["description"] = description
        self.datasets[name] = dataset
        return dataset

    def flush(self) -> None:
        if not self.buffer or not self.buffer.get("time_s"):
            return
        frame_count = len(self.buffer["time_s"])
        for name, samples in self.buffer.items():
            stacked = np.stack(samples, axis=0)
            dataset = self._dataset_for(name, stacked[0])
            start = dataset.shape[0]
            dataset.resize((start + frame_count,) + dataset.shape[1:])
            dataset[start:start + frame_count] = stacked
        self.buffer.clear()
        self.h5_file.flush()


def _make_case_cfg(base_cfg: HexClimbCfg, gravity: Sequence[float],
                   suction_max: float, suction_delta: float) -> HexClimbCfg:
    cfg = copy.deepcopy(base_cfg)
    cfg.env.num_envs = len(MASS_SCALES)
    cfg.env.enable_recording_sensors = True
    cfg.env.record_mass_scales = list(MASS_SCALES)
    cfg.env.record_fixed_friction = FIXED_ROBOT_SHAPE_FRICTION
    cfg.env.visualize_gravity = True
    cfg.env.episode_length_s = max(
        cfg.env.episode_length_s,
        SETTLE_DURATION_S + len(_command_segments(cfg))
        * (TRANSITION_DURATION_S + RECORD_DURATION_S) + 5.0,
    )
    cfg.sim.gravity = list(gravity)
    cfg.terrain.static_friction = FIXED_STATIC_FRICTION
    cfg.terrain.dynamic_friction = FIXED_DYNAMIC_FRICTION
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False
    cfg.noise.add_noise = False
    cfg.commands.resampling_time = 1.0e9
    cfg.control.suction_force_max = float(suction_max)
    cfg.control.suction_force_delt = float(suction_delta)
    return cfg


def _set_command(env, command: torch.Tensor) -> None:
    env.commands[:, :3] = command.view(1, 3)


def _advance_expert(env, command: torch.Tensor, callback=None):
    _set_command(env, command)
    q_des, tau_ff, adhesions = env.get_expert_commands()
    return env.step_q_tao(q_des, tau_ff, adhesions, physics_step_callback=callback)


def _verify_friction(env) -> np.ndarray:
    all_friction = []
    for env_handle, actor_handle in zip(env.envs, env.actor_handles):
        props = env.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
        all_friction.append([shape.friction for shape in props])
    friction = np.asarray(all_friction, dtype=np.float32)
    if not np.allclose(friction, FIXED_ROBOT_SHAPE_FRICTION, atol=1e-6):
        raise RuntimeError(
            "robot rigid-shape friction does not match the requested fixed value"
        )
    return friction


def _case_group_name(angle_deg: float, suction_max: float, suction_delta: float) -> str:
    return (
        f"gravity_x_{angle_deg:+.3f}deg/"
        f"suction_max_{suction_max:.3f}_delta_{suction_delta:.3f}"
    )


def _record_case(h5_file, env, angle_deg: float, gravity: Sequence[float],
                 suction_max: float, suction_delta: float) -> Dict[str, Any]:
    import h5py

    group_name = _case_group_name(angle_deg, suction_max, suction_delta)
    group = h5_file.create_group(group_name)
    group.attrs["gravity_world_m_s2"] = np.asarray(gravity, dtype=np.float64)
    group.attrs["gravity_x_angle_deg"] = angle_deg
    group.attrs["suction_force_max_N"] = suction_max
    group.attrs["suction_force_delta_N_per_control_step"] = suction_delta
    group.attrs["physical_dt_s"] = env.cfg.sim.dt
    group.attrs["control_dt_s"] = env.dt
    group.attrs["physical_sample_rate_hz"] = 1.0 / env.cfg.sim.dt
    group.attrs["joint_sensor_labels_json"] = json.dumps(env.record_joint_sensor_specs)
    group.attrs["wrench_order"] = "[Fx,Fy,Fz,Mx,My,Mz]"

    writer = H5CaseWriter(h5_file, group, env.num_envs, H5_CHUNK_FRAMES)
    writer.write_static("total_mass_kg", env.record_total_masses.detach().cpu().numpy(), unit="kg")
    writer.write_static("rigid_body_masses_kg", env.record_body_masses.detach().cpu().numpy(), unit="kg")
    writer.write_static("rigid_body_com_local_m", env.record_body_com_local.detach().cpu().numpy(), unit="m")
    writer.write_static("rigid_body_names", np.asarray(env.body_names, dtype=h5py.string_dtype()))
    writer.write_static("dof_names", np.asarray(env.dof_names, dtype=h5py.string_dtype()))
    motor_dof_names = [
        env.dof_names[index]
        for index in env.dof_motor_drive_indices.detach().cpu().tolist()
    ]
    writer.write_static(
        "motor_dof_names", np.asarray(motor_dof_names, dtype=h5py.string_dtype())
    )
    writer.write_static("robot_shape_friction", _verify_friction(env), unit="coefficient")
    writer.write_static(
        "surface_friction",
        np.asarray([FIXED_STATIC_FRICTION, FIXED_DYNAMIC_FRICTION], dtype=np.float32),
        labels="[static,dynamic]", unit="coefficient",
    )
    cup_labels = [env.body_names[int(index)] for index in env.record_cup_reference_indices.cpu().tolist()]
    writer.write_static("cup_reference_body_names", np.asarray(cup_labels, dtype=h5py.string_dtype()))

    segments = _command_segments(env.cfg)
    writer.write_static("segment_names", np.asarray([name for name, _ in segments], dtype=h5py.string_dtype()))
    writer.write_static(
        "segment_target_command",
        np.asarray([command for _, command in segments], dtype=np.float32),
        labels="[vx_m_s,vy_m_s,yaw_rate_rad_s]",
    )

    # Attach cups and allow the expert state/contact filters to settle.  These
    # frames are deliberately not recorded.
    zero_command = torch.zeros(3, dtype=torch.float32, device=env.device)
    for _ in range(int(round(SETTLE_DURATION_S / env.dt))):
        _, _, _, dones, _ = _advance_expert(env, zero_command)
        if dones.any():
            raise RuntimeError("environment reset during the unrecorded settle phase")

    physical_dt = float(env.cfg.sim.dt)
    transition_steps = int(round(TRANSITION_DURATION_S / env.dt))
    record_steps = int(round(RECORD_DURATION_S / env.dt))
    elapsed_recorded_s = 0.0
    previous = zero_command

    for segment_id, (_, target_values) in enumerate(segments):
        target = torch.tensor(target_values, dtype=torch.float32, device=env.device)
        for step_index in range(transition_steps):
            alpha = float(step_index + 1) / max(transition_steps, 1)
            command = previous + alpha * (target - previous)
            _, _, _, dones, _ = _advance_expert(env, command)
            if dones.any():
                raise RuntimeError(f"environment reset during transition for segment {segment_id}")

        def record_callback(current_env, segment_id=segment_id):
            nonlocal elapsed_recorded_s
            writer.append(
                current_env.get_recording_telemetry(), elapsed_recorded_s, segment_id
            )
            elapsed_recorded_s += physical_dt

        for _ in range(record_steps):
            _, _, _, dones, _ = _advance_expert(env, target, record_callback)
            if dones.any():
                raise RuntimeError(f"environment reset during recorded segment {segment_id}")
        previous = target

    writer.flush()
    expected_frames = len(segments) * record_steps * env.cfg.control.decimation
    actual_frames = writer.datasets["time_s"].shape[0]
    if actual_frames != expected_frames:
        raise RuntimeError(f"recorded {actual_frames} frames, expected {expected_frames}")
    return {
        "group": group_name,
        "gravity_world_m_s2": list(gravity),
        "mass_scales": list(MASS_SCALES),
        "total_mass_kg": env.record_total_masses.detach().cpu().tolist(),
        "frames": actual_frames,
    }


def _destroy_env(env) -> None:
    if getattr(env, "viewer", None) is not None:
        env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)


def main() -> None:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("record_expert_climb.py requires h5py in the Isaac Gym environment") from exc

    if not MASS_SCALES or any(scale <= 0.0 for scale in MASS_SCALES):
        raise ValueError("MASS_SCALES must contain only positive values")
    if GRAVITY_MAGNITUDE < 0.0:
        raise ValueError("GRAVITY_MAGNITUDE must be non-negative")

    args = get_args()
    # The number of environments is defined by the mass sweep, not by a CLI
    # override, so all recorded fields have a stable mass_scale dimension.
    args.num_envs = None
    base_cfg, _ = task_registry.get_cfgs("hex_climb")
    run_dir = _make_run_dir()
    if not README_TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"README template is missing: {README_TEMPLATE_PATH}")
    shutil.copyfile(README_TEMPLATE_PATH, run_dir / "README.md")

    run_config = {
        "dataset_version": DATASET_VERSION,
        "manual_experiment_configuration": {
            "gravity_x_angles_deg": list(GRAVITY_X_ANGLES_DEG),
            "gravity_magnitude_m_s2": GRAVITY_MAGNITUDE,
            "mass_scales": list(MASS_SCALES),
            "suction_force_maxes_N": list(SUCTION_FORCE_MAXES),
            "suction_force_deltas_N_per_control_step": list(SUCTION_FORCE_DELTAS),
            "fixed_static_friction": FIXED_STATIC_FRICTION,
            "fixed_dynamic_friction": FIXED_DYNAMIC_FRICTION,
            "fixed_robot_shape_friction": FIXED_ROBOT_SHAPE_FRICTION,
            "transition_duration_s": TRANSITION_DURATION_S,
            "record_duration_s": RECORD_DURATION_S,
            "settle_duration_s": SETTLE_DURATION_S,
        },
        "base_hex_climb_config": _as_jsonable(class_to_dict(base_cfg)),
    }
    # JSON is a valid YAML subset and avoids adding a second runtime dependency.
    (run_dir / "config_snapshot.yaml").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metadata = {
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git_revision": _git_revision(),
        "urdf": HexClimbCfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
        "physical_sample_rate_hz": 1.0 / HexClimbCfg.sim.dt,
        "control_rate_hz": 1.0 / (HexClimbCfg.sim.dt * HexClimbCfg.control.decimation),
        "cases": [],
    }
    metadata_path = run_dir / "run_metadata.json"

    try:
        with h5py.File(run_dir / "expert_climb.h5", "w") as h5_file:
            h5_file.attrs["dataset_version"] = DATASET_VERSION
            h5_file.attrs["wrench_order"] = "[Fx,Fy,Fz,Mx,My,Mz]"
            h5_file.attrs["mass_dimension"] = "MASS_SCALES order"
            mass_scales = h5_file.create_dataset(
                "mass_scales", data=np.asarray(MASS_SCALES, dtype=np.float32)
            )
            mass_scales.attrs["unit"] = "scale"
            mass_scales.attrs["description"] = (
                "Mass scale for every dynamic dataset's second dimension; "
                "shared by all cases in this run."
            )
            for angle_deg in GRAVITY_X_ANGLES_DEG:
                gravity = _gravity_from_x_angle(float(angle_deg))
                for suction_max in SUCTION_FORCE_MAXES:
                    for suction_delta in SUCTION_FORCE_DELTAS:
                        case_cfg = _make_case_cfg(
                            base_cfg, gravity, float(suction_max), float(suction_delta)
                        )
                        env = task_registry.make_env(
                            name="hex_climb", args=args, env_cfg=case_cfg
                        )[0]
                        try:
                            env.reset()
                            case_metadata = _record_case(
                                h5_file, env, float(angle_deg), gravity,
                                float(suction_max), float(suction_delta),
                            )
                            metadata["cases"].append(case_metadata)
                            h5_file.flush()
                        finally:
                            _destroy_env(env)
    finally:
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(f"Expert climb recording completed: {run_dir}")


if __name__ == "__main__":
    main()
