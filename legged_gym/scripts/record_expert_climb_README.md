# Expert Climb HDF5 recording

`record_expert_climb.py` copies this file to each run directory:

```text
logs/expert_climb/<run_id>/
├── expert_climb.h5
├── README.md
├── config_snapshot.yaml
└── run_metadata.json
```

## HDF5 layout

`expert_climb.h5` stores the shared `mass_scales` dataset once at its root. It
contains one group per `gravity_x_angle × suction_max × suction_delta` case.
Dynamic robot fields are in `data/` and use:

```text
[time, mass_scale, ...]
```

`mass_scale` follows the root-level `mass_scales` dataset exactly. Wrenches use force-first order `[Fx, Fy, Fz, Mx, My, Mz]`.

## Dynamic dataset dictionary

- `joint_position` — rad; all URDF DOF positions. See group-level `dof_names` for ordering.
- `joint_velocity` — rad/s; all URDF DOF velocities.
- `actuator_torque` — N m; torque submitted to PhysX after controller calculation and clipping.
- `feedforward_torque` — N m; expert main-motor feedforward torque before it is added to position-control torque and before total actuator clipping. See group-level `motor_dof_names` for ordering.
- `dof_generalized_force` — N m; Isaac Gym DOF force-sensor reading, including generalized constraint effects.
- `joint_sensor_wrench_world_raw` — N, N m; raw 6D sensor wrench in world axes at the child-link origin.
- `joint_sensor_origin_world` — m; sensor origin in world coordinates.
- `joint_parent_pose_world` — `[x,y,z,qx,qy,qz,qw]`; parent pose used for the wrench conversion.
- `joint_wrench_parent` — N, N m; wrench at the parent-link origin in current parent-link axes. Positive means the downstream assembly acts on its parent.
- `com_velocity_world` — m/s; mass-weighted true whole-robot centre-of-mass velocity in world axes.
- `com_velocity_body` — m/s; the same velocity expressed in the root/body frame.
- `desired_velocity_body` — `[vx,vy,yaw_rate]`, with units `[m/s,m/s,rad/s]`.
- `adhesion_command_wrench_world` — N, N m; three applied adhesion forces combined about the toe/cup reference point.
- `surface_force_world` — N; net environment force on the `suck/toe/empty` bodies. Its negative is the robot force on the surface.
- `cup_interface_wrench_parent` — N, N m; `foot → ball1` interface load, at the foot origin in current foot axes. Positive means the cup assembly acts on the foot.
- `cup_reference_pose_world` — `[x,y,z,qx,qy,qz,qw]`; toe-body pose used as the cup reference.
- `time_s` — s; elapsed recorded physical simulation time in the current case. Settle and command-transition samples are excluded.
- `segment_id` — index into group-level `segment_names` and `segment_target_command`.
- `sample_valid` — bool; one for every saved stable sample.

## Static case metadata

Every case group stores gravity, suction parameters, friction values, final total and per-link masses, link local COM values, body/DOF names, `motor_dof_names`, sensor labels, cup labels and all command segments. HDF5 dataset attributes repeat units and short descriptions.

`config_snapshot.yaml` is JSON-formatted YAML and records the script configuration. `run_metadata.json` records creation time, Git revision, URDF path, sampling rates and completed cases.

## Force interpretation

`surface_force_world` is a resultant collision force only. Isaac Gym’s net-contact tensor does not expose a reliable contact point or exact resultant contact moment for this plane contact. Do not interpret it as an exact surface 6D wrench. For cup selection, evaluate it together with `adhesion_command_wrench_world` and `cup_interface_wrench_parent`.

The record callback runs after each `gym.simulate()` call (`sim.dt = 0.0025 s` by default), i.e. 400 Hz. It does not run after each internal PhysX `substeps` iteration.
