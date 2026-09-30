"""Plot motor torque and joint wrench data from an expert-climb mass sweep.

The configuration lists below are intended to be edited before running.  For
example, set ``LEGS_TO_PLOT = ["rf", "rm"]`` to plot two legs, or limit
``JOINTS_TO_PLOT`` to ``["knee"]``.  The HDF5 metadata, rather than a fixed
DOF ordering, is used to resolve each requested joint.

Run with the same Python environment used to record the data, for example:

    /home/val/miniconda3/envs/env_gym/bin/python \
        legged_gym/scripts/plot_expert_climb_mass_sweep.py

The script produces one independent torque figure for each configured value in
``TORQUE_TYPES_TO_PLOT`` and one 6-axis wrench figure per selected leg.  A wrench is ``[Fx, Fy, Fz, Mx, My, Mz]`` from
``joint_wrench_parent``: it is expressed at the parent-link origin in the
current parent-link axes, and positive means the downstream assembly acts on
its parent.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Edit these lists to choose what is plotted.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = PROJECT_ROOT / "logs/expert_climb/20260930_185902_mass_sweep"

# Empty list means every complete case found in expert_climb.h5.  The selected
# default is the complete 180-degree case in this recording (48,000 frames).
CASE_PATHS: List[str] = [
    "gravity_x_+0.000deg/suction_max_50.000_delta_50.000",
    # "gravity_x_+60.000deg/suction_max_300.000_delta_30.000",
    # "gravity_x_+90.000deg/suction_max_300.000_delta_30.000"
]

# Leg prefixes and joint suffixes.  Valid examples are rf, rm, rb, lf, lm, lb
# and thigh, knee, ankle, foot, ball.  The default selects the three RM motors.
LEGS_TO_PLOT: List[str] = ["lm"]
JOINTS_TO_PLOT: List[str] = ["thigh", "knee", "ankle"]

# Empty list plots every recorded scale.  Otherwise, list exact scale values,
# for example ``[0.5, 0.8, 1.0]``.
MASS_SCALES_TO_PLOT: List[float] = [0.2,0.5,1.0]

# Inclusive recorded-time interval in seconds.  Use ``(None, None)`` for the
# complete recording, ``(0.0, 30.0)`` for its first 30 seconds, or leave one
# endpoint as None for an open-ended interval.
TIME_RANGE_S: Tuple[Optional[float], Optional[float]] = (None,None)

# Plot at most this many samples per trace.  The recording remains untouched;
# this only thins samples for faster, more legible raster figures.
MAX_PLOT_SAMPLES = 6_000
PLOT_DIR_NAME = "plots_mass_sweep"

# Select independent figures for actual actuator torque, expert feedforward
# torque, or both. Valid values are "real" and "feedforward".
TORQUE_TYPES_TO_PLOT: List[str] = ["real", "feedforward"]
TORQUE_PLOT_SPECS: Dict[str, Tuple[str, str, str]] = {
    "real": (
        "actuator_torque",
        "real actuator torque",
        "real actuator torque",
    ),
    "feedforward": (
        "feedforward_torque",
        "feedforward torque",
        "feedforward torque",
    ),
}


FORCE_LABELS = ("Fx", "Fy", "Fz")
MOMENT_LABELS = ("Mx", "My", "Mz")


def _decode(values: Iterable[object]) -> List[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _strip_prefix(value: str, prefix: str) -> str:
    """Python 3.8-compatible equivalent of ``str.removeprefix``."""
    return value[len(prefix):] if value.startswith(prefix) else value


def _case_paths(h5_file: h5py.File) -> List[str]:
    """Return paths for leaf groups which contain dynamic recording data."""
    paths: List[str] = []

    def visit(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Group) and "data" in obj and "time_s" in obj["data"]:
            paths.append(name)

    h5_file.visititems(visit)
    return paths


def _select_cases(h5_file: h5py.File, requested: Sequence[str]) -> List[str]:
    available = _case_paths(h5_file)
    if not requested:
        return available
    missing = [path for path in requested if path not in available]
    if missing:
        raise KeyError(
            "Requested case(s) not found: " + ", ".join(missing)
            + "\nAvailable cases:\n  " + "\n  ".join(available)
        )
    return list(requested)


def _sample_stride(frame_count: int) -> int:
    return max(1, math.ceil(frame_count / MAX_PLOT_SAMPLES))


def _selected_torque_types() -> List[str]:
    """Validate the configured independent torque figure types."""
    selected = list(TORQUE_TYPES_TO_PLOT)
    if not selected:
        raise ValueError("TORQUE_TYPES_TO_PLOT must select at least one torque type.")
    if len(selected) != len(set(selected)):
        raise ValueError("TORQUE_TYPES_TO_PLOT must not contain duplicate values.")
    invalid = [value for value in selected if value not in TORQUE_PLOT_SPECS]
    if invalid:
        allowed = ", ".join(TORQUE_PLOT_SPECS)
        raise ValueError(
            "TORQUE_TYPES_TO_PLOT contains unsupported value(s): "
            + ", ".join(repr(value) for value in invalid)
            + f". Allowed values: {allowed}"
        )
    return selected


def _selected_mass_indices(mass_scales: np.ndarray) -> List[int]:
    """Return configured mass-scale indices, preserving requested order."""
    if not MASS_SCALES_TO_PLOT:
        return list(range(len(mass_scales)))

    selected: List[int] = []
    for requested_scale in MASS_SCALES_TO_PLOT:
        matches = np.flatnonzero(np.isclose(mass_scales, requested_scale, rtol=1e-6, atol=1e-6))
        if len(matches) != 1:
            available = ", ".join(f"{scale:g}" for scale in mass_scales)
            raise ValueError(
                f"Requested mass scale {requested_scale:g} is unavailable or ambiguous. "
                f"Available scales: {available}"
            )
        index = int(matches[0])
        if index not in selected:
            selected.append(index)
    return selected


def _selected_time_slice(time_s: np.ndarray) -> Tuple[slice, int]:
    """Apply TIME_RANGE_S, then choose a display stride within that interval."""
    start_s, end_s = TIME_RANGE_S
    if start_s is not None and end_s is not None and start_s > end_s:
        raise ValueError(f"TIME_RANGE_S start ({start_s}) must not exceed end ({end_s}).")

    start_index = 0 if start_s is None else int(np.searchsorted(time_s, start_s, side="left"))
    stop_index = len(time_s) if end_s is None else int(np.searchsorted(time_s, end_s, side="right"))
    if start_index >= stop_index:
        raise ValueError(
            f"TIME_RANGE_S={TIME_RANGE_S} contains no samples; recording spans "
            f"{time_s[0]:.4f} to {time_s[-1]:.4f} s."
        )
    stride = _sample_stride(stop_index - start_index)
    return slice(start_index, stop_index, stride), stride


def _selected_joint_names(
    dof_names: Sequence[str], leg: str, joint_suffixes: Sequence[str],
) -> List[Tuple[str, int]]:
    """Resolve configured suffixes to motor DOF names and indices."""
    name_to_index = {name.lower(): index for index, name in enumerate(dof_names)}
    selected: List[Tuple[str, int]] = []
    for suffix in joint_suffixes:
        suffix = suffix.lower()
        full_name = suffix if suffix.startswith("j_") else f"j_{leg}_{suffix}"
        index = name_to_index.get(full_name)
        if index is None:
            raise KeyError(
                f"Motor DOF {full_name!r} is absent. Available DOFs include: "
                + ", ".join(dof_names)
            )
        selected.append((full_name, index))
    return selected


def _sensor_index(sensor_specs: Sequence[dict], leg: str, joint_suffix: str) -> int:
    target = f"{leg}_{_strip_prefix(joint_suffix, 'j_' + leg + '_')}".lower()
    for index, spec in enumerate(sensor_specs):
        if str(spec["label"]).lower() == target:
            return index
    labels = ", ".join(str(spec["label"]) for spec in sensor_specs)
    raise KeyError(f"Joint-wrench sensor {target!r} is absent. Available sensors: {labels}")


def _mass_labels(mass_scales: np.ndarray, total_masses: np.ndarray) -> List[str]:
    return [
        f"scale {scale:g} ({mass:.2f} kg)"
        for scale, mass in zip(mass_scales, total_masses)
    ]


def _add_segment_boundaries(ax: plt.Axes, time_s: np.ndarray, segment_id: np.ndarray) -> None:
    """Mark changes of velocity-command segment without obscuring traces."""
    change_indices = np.flatnonzero(np.diff(segment_id) != 0) + 1
    for index in change_indices:
        ax.axvline(time_s[index], color="0.60", linewidth=0.55, alpha=0.65, zorder=0)


def _case_label(case: h5py.Group) -> str:
    angle = float(case.attrs["gravity_x_angle_deg"])
    suction = float(case.attrs["suction_force_max_N"])
    return f"gravity x = {angle:g}°, suction max = {suction:g} N"


def _safe_stem(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").replace("+", "plus")


def _plot_torque(
    case: h5py.Group,
    output_dir: Path,
    leg: str,
    selected_dofs: Sequence[Tuple[str, int]],
    time_s: np.ndarray,
    segment_id: np.ndarray,
    mass_labels: Sequence[str],
    mass_indices: Sequence[int],
    frame_slice: slice,
    torque_type: str,
) -> Path:
    """Plot one selected torque source for the configured leg and DOFs."""
    dataset_name, torque_label, title_label = TORQUE_PLOT_SPECS[torque_type]
    data = case["data"]
    fig, axes = plt.subplots(
        len(selected_dofs), 1, sharex=True, figsize=(13, 3.0 * len(selected_dofs)),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(mass_labels)))

    for axis, (dof_name, dof_index) in zip(axes, selected_dofs):
        # Shape after striding: [time, mass_scale].
        torque = data[dataset_name][frame_slice, :, dof_index][:, mass_indices]
        for mass_index, label in enumerate(mass_labels):
            axis.plot(time_s, torque[:, mass_index], color=colors[mass_index],
                      linewidth=0.8, label=label)
        _add_segment_boundaries(axis, time_s, segment_id)
        axis.set_ylabel(f"{dof_name}\n{torque_label} [N m]")
        axis.grid(True, alpha=0.25)

    axes[0].legend(loc="upper right", ncol=2, fontsize=8, frameon=True)
    axes[-1].set_xlabel("recorded simulation time [s]")
    fig.suptitle(
        f"{leg.upper()} {title_label} across mass scales — {_case_label(case)}\n"
        "vertical lines: velocity-command segment boundaries"
    )
    path = output_dir / f"{leg}_{torque_type}_torque.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_joint_wrench(
    case: h5py.Group,
    output_dir: Path,
    leg: str,
    joint_suffix: str,
    sensor_index: int,
    time_s: np.ndarray,
    segment_id: np.ndarray,
    mass_labels: Sequence[str],
    mass_indices: Sequence[int],
    frame_slice: slice,
    stride: int,
) -> Path:
    """Plot all three force and moment components for one joint sensor."""
    # Shape: [time, mass_scale, Fx/Fy/Fz/Mx/My/Mz].
    wrench = case["data/joint_wrench_parent"][frame_slice, :, sensor_index, :][:, mass_indices]
    fig, axes = plt.subplots(2, 3, sharex=True, figsize=(16, 7.4), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(mass_labels)))
    component_labels = FORCE_LABELS + MOMENT_LABELS
    component_units = ("N", "N", "N", "N m", "N m", "N m")

    for component, axis in enumerate(axes.flat):
        for mass_index, label in enumerate(mass_labels):
            axis.plot(time_s, wrench[:, mass_index, component], color=colors[mass_index],
                      linewidth=0.8, label=label)
        _add_segment_boundaries(axis, time_s, segment_id)
        axis.set_title(f"{component_labels[component]} [{component_units[component]}]")
        axis.grid(True, alpha=0.25)

    for axis in axes[-1]:
        axis.set_xlabel("recorded simulation time [s]")
    axes[0, 0].legend(loc="upper right", ncol=2, fontsize=8, frameon=True)
    fig.suptitle(
        f"{leg.upper()} {joint_suffix} joint wrench across mass scales — {_case_label(case)}\n"
        "joint_wrench_parent: parent-link axes, parent-link origin; positive = downstream assembly acting on parent"
    )
    path = output_dir / f"{leg}_{joint_suffix}_joint_wrench_parent.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_case(
    case: h5py.Group,
    output_dir: Path,
    mass_scales: np.ndarray,
    torque_types: Sequence[str],
) -> List[Path]:
    data = case["data"]
    required_case_datasets = ("total_mass_kg", "dof_names", "motor_dof_names")
    missing_case_datasets = [name for name in required_case_datasets if name not in case]
    if missing_case_datasets:
        raise KeyError(
            f"Case {case.name} is missing required dataset(s): "
            + ", ".join(missing_case_datasets)
        )
    required_data_datasets = (
        "time_s", "segment_id", "actuator_torque", "feedforward_torque",
        "joint_wrench_parent",
    )
    missing_data_datasets = [name for name in required_data_datasets if name not in data]
    if missing_data_datasets:
        raise KeyError(
            f"Case {case.name} is missing required data dataset(s): "
            + ", ".join(missing_data_datasets)
        )

    raw_time_s = data["time_s"][:]
    frame_slice, stride = _selected_time_slice(raw_time_s)
    time_s = raw_time_s[frame_slice]
    segment_id = data["segment_id"][frame_slice]
    total_masses = case["total_mass_kg"][:]
    dof_names = _decode(case["dof_names"][:])
    motor_dof_names = _decode(case["motor_dof_names"][:])
    if total_masses.shape != mass_scales.shape:
        raise ValueError(
            f"Case {case.name} total_mass_kg shape {total_masses.shape} does not match "
            f"root mass_scales shape {mass_scales.shape}."
        )

    expected_time_count = len(raw_time_s)
    torque_shapes = {
        "actuator_torque": len(dof_names),
        "feedforward_torque": len(motor_dof_names),
    }
    for dataset_name, expected_dof_count in torque_shapes.items():
        shape = data[dataset_name].shape
        if (len(shape) != 3 or shape[0] != expected_time_count
                or shape[1] != len(mass_scales) or shape[2] != expected_dof_count):
            raise ValueError(
                f"{case.name}/data/{dataset_name} has shape {shape}; expected "
                f"({expected_time_count}, {len(mass_scales)}, {expected_dof_count})."
            )

    mass_indices = _selected_mass_indices(mass_scales)
    mass_labels = _mass_labels(mass_scales[mass_indices], total_masses[mass_indices])
    sensor_specs = json.loads(case.attrs["joint_sensor_labels_json"])
    motor_name_to_index = {name.lower(): index for index, name in enumerate(motor_dof_names)}

    print(
        f"  selected frames={len(time_s)}, time={time_s[0]:.4f}..{time_s[-1]:.4f} s, "
        f"plot stride={stride}, masses: " + ", ".join(mass_labels)
    )
    written: List[Path] = []
    for leg in LEGS_TO_PLOT:
        leg = leg.lower()
        selected_dofs = _selected_joint_names(dof_names, leg, JOINTS_TO_PLOT)
        if "real" in torque_types:
            written.append(_plot_torque(
                case, output_dir, leg, selected_dofs, time_s, segment_id, mass_labels,
                mass_indices, frame_slice, "real",
            ))
        if "feedforward" in torque_types:
            feedforward_dofs = []
            missing_feedforward_dofs = []
            for dof_name, _ in selected_dofs:
                motor_index = motor_name_to_index.get(dof_name.lower())
                if motor_index is None:
                    missing_feedforward_dofs.append(dof_name)
                else:
                    feedforward_dofs.append((dof_name, motor_index))
            if missing_feedforward_dofs:
                print(
                    "  skipping feedforward torque for non-motor DOF(s): "
                    + ", ".join(missing_feedforward_dofs)
                )
            if feedforward_dofs:
                written.append(_plot_torque(
                    case, output_dir, leg, feedforward_dofs, time_s, segment_id,
                    mass_labels, mass_indices, frame_slice, "feedforward",
                ))
        for dof_name, _ in selected_dofs:
            # j_rm_thigh -> thigh; sensor labels do not include the j_ prefix.
            joint_suffix = _strip_prefix(dof_name, f"j_{leg}_")
            sensor_index = _sensor_index(sensor_specs, leg, joint_suffix)
            written.append(_plot_joint_wrench(
                case, output_dir, leg, joint_suffix, sensor_index, time_s, segment_id,
                mass_labels, mass_indices, frame_slice, stride,
            ))
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR,
                        help="Directory containing expert_climb.h5.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: <run-dir>/" + PLOT_DIR_NAME + ").")
    parser.add_argument("--case", action="append", dest="cases", default=None,
                        help="HDF5 case path. Repeat to select multiple cases; overrides CASE_PATHS.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torque_types = _selected_torque_types()
    h5_path = args.run_dir / "expert_climb.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(f"Recording file does not exist: {h5_path}")
    output_root = args.output_dir or args.run_dir / PLOT_DIR_NAME

    with h5py.File(h5_path, "r") as h5_file:
        if "mass_scales" not in h5_file:
            raise KeyError("Recording is missing the required root-level mass_scales dataset.")
        mass_scales = h5_file["mass_scales"][:]
        if mass_scales.ndim != 1 or len(mass_scales) == 0:
            raise ValueError("Root mass_scales must be a non-empty one-dimensional dataset.")
        if not np.isfinite(mass_scales).all() or np.any(mass_scales <= 0.0):
            raise ValueError("Root mass_scales must contain only finite positive values.")

        requested_cases = CASE_PATHS if args.cases is None else args.cases
        selected_cases = _select_cases(h5_file, requested_cases)
        if not selected_cases:
            raise RuntimeError("No cases selected. Set CASE_PATHS or pass --case.")
        for case_path in selected_cases:
            print(f"Plotting {case_path}")
            case_output_dir = output_root / _safe_stem(case_path)
            case_output_dir.mkdir(parents=True, exist_ok=True)
            written = _plot_case(
                h5_file[case_path], case_output_dir, mass_scales, torque_types
            )
            for path in written:
                print(f"  wrote {path}")


if __name__ == "__main__":
    main()
