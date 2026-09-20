# Agent Task: Open3D Keyboard Teleop Pose Labeler

## Goal

Implement a first-version interactive Open3D sandbox for manually controlling and labeling robot body poses in a point-cloud environment.

The tool should let a human operator use the keyboard to move the robot body pose in SE(3), see immediate visual feedback about whether the pose is feasible, and save demonstration trajectories for later imitation learning or planner debugging.

This is **not** a full planner, not a true haptic controller, and not a real robot control loop. It is an offline/interactive pose-control and data-collection tool.

---

## Context

The project already has feasibility checking logic in:

- `env_robot_voxels.py`
- `se3_planning.py`

Important existing method:

```python
HexState.RobotFeasiCheck(W_T_R)
```

It returns:

```python
idx, possible_landing_idx, body_cf_mask, leg_point_feasi_mask
```

where `leg_point_feasi_mask` has shape approximately:

```python
(6, landing_num, 2)
```

representing feasibility for 6 legs and 2 IK branches.

The current strong planner validity logic in `se3_planning.py` checks approximately:

```python
body_valid = not (~body_cf_mask).any()
leg_counts = leg_point_feasi_mask.any(axis=-1).sum(axis=1)
strong_valid = body_valid and (leg_counts >= 6).all()
```

For this teleop tool, use a weaker validity level as well:

```python
weak_valid = body_valid and (leg_counts >= 1).all()
strong_valid = body_valid and (leg_counts >= 6).all()
```

---

## Deliverable

Create a script, preferably:

```text
keyboard_pose_labeler.py
```

The script should:

1. Open an Open3D window.
2. Display the static environment point cloud and landing points.
3. Display a movable robot body pose as a simple body box and coordinate frame.
4. Let the user move the robot body pose with keyboard commands.
5. On each pose update, call `RobotFeasiCheck`.
6. Color the robot body according to feasibility.
7. Display or print useful feasibility information.
8. Allow the user to save poses into a trajectory.
9. Allow exporting the trajectory as JSON.

---

## Recommended Implementation

Use:

```python
open3d.visualization.VisualizerWithKeyCallback
```

Keep the Open3D window alive and update geometries in-place:

```python
vis.update_geometry(geometry)
vis.poll_events()
vis.update_renderer()
```

Do **not** recreate the whole window after every key press.

---

## Main Components

### 1. PoseTeleopState

Create a class to store teleop state:

```python
class PoseTeleopState:
    def __init__(self):
        self.W_T_R = np.eye(4)
        self.prev_W_T_R = np.eye(4)
        self.translation_step = 0.03
        self.rotation_step = np.deg2rad(5.0)
        self.saved_poses = []
        self.last_eval = None
        self.show_mode = "validity"
```

The pose convention should be a 4x4 transform matrix.

---

### 2. Keyboard Controls

Use local body-frame increments.

Recommended controls:

| Key | Action |
|---|---|
| `W` | move forward along body +Y |
| `S` | move backward along body -Y |
| `A` | move left along body -X |
| `D` | move right along body +X |
| `R` | move up along body +Z |
| `F` | move down along body -Z |
| `J` | yaw left |
| `L` | yaw right |
| `I` | pitch up |
| `K` | pitch down |
| `U` | roll left |
| `O` | roll right |
| `Z` | decrease translation/rotation step |
| `X` | increase translation/rotation step |
| `Space` | save current pose to trajectory |
| `Backspace` | delete last saved pose |
| `Enter` | export trajectory JSON |
| `C` | cycle visualization mode |
| `Q` or `Esc` | close window |

If Open3D key callback has trouble with special keys like Space, Enter, or Backspace, use ordinary letter alternatives:

| Key | Backup Action |
|---|---|
| `P` | save current pose |
| `B` | delete last saved pose |
| `E` | export trajectory |

---

### 3. SE3 Increment Helpers

Implement local transform helpers:

```python
def make_translation(dx, dy, dz):
    T = np.eye(4)
    T[:3, 3] = [dx, dy, dz]
    return T


def rot_x(theta): ...
def rot_y(theta): ...
def rot_z(theta): ...


def make_rotation(axis, theta):
    # return 4x4 local rotation transform
```

When applying a local body-frame command:

```python
state.W_T_R = state.W_T_R @ delta_T
```

This means movement is in robot/body coordinates.

---

### 4. Open3D Geometry

Display at least:

1. Environment point cloud.
2. Landing points if available.
3. Robot body box.
4. Robot body coordinate frame.
5. Saved trajectory line.
6. Optional: feasible landing points for each leg.

For the body, use a simple box mesh first:

```python
body_mesh = o3d.geometry.TriangleMesh.create_box(width, height, depth)
```

Center it at the robot frame before applying transform:

```python
body_mesh.translate([-width / 2, -height / 2, -depth / 2])
```

Then transform it with `W_T_R`.

To update body pose, keep the mesh object and apply relative transform:

```python
dT = new_W_T_R @ np.linalg.inv(old_W_T_R)
body_mesh.transform(dT)
vis.update_geometry(body_mesh)
```

For the coordinate frame, either:

- remove and recreate the frame on each update, or
- implement it as a `LineSet` and update its points.

First version can remove/recreate the coordinate frame because it is small.

---

### 5. Feasibility Evaluation

Implement:

```python
def evaluate_pose(hex_state, W_T_R):
    idx, possible_landing_idx, body_cf_mask, leg_mask = hex_state.RobotFeasiCheck(W_T_R)

    body_valid = not (~body_cf_mask).any()
    leg_counts = leg_mask.any(axis=-1).sum(axis=1)

    weak_valid = body_valid and (leg_counts >= 1).all()
    strong_valid = body_valid and (leg_counts >= 6).all()

    if not body_valid:
        score = -100.0
    else:
        score = float(leg_counts.min() + 0.05 * leg_counts.sum())

    return {
        "body_valid": bool(body_valid),
        "weak_valid": bool(weak_valid),
        "strong_valid": bool(strong_valid),
        "leg_counts": leg_counts.astype(int).tolist(),
        "min_leg_count": int(leg_counts.min()) if len(leg_counts) else 0,
        "total_leg_count": int(leg_counts.sum()),
        "score": score,
        "idx": idx,
        "possible_landing_idx": possible_landing_idx,
        "leg_mask": leg_mask,
    }
```

Use this result to color the robot body:

```text
strong_valid: green
weak_valid: yellow
invalid: red
```

Suggested colors:

```python
GREEN = [0.1, 0.8, 0.1]
YELLOW = [0.9, 0.7, 0.1]
RED = [0.9, 0.1, 0.1]
GRAY = [0.5, 0.5, 0.5]
```

---

### 6. Terminal Feedback

On every key update, print one compact line:

```text
pose: x=..., y=..., z=..., rpy=(..., ..., ...), valid=strong/weak/invalid, legs=[...], score=...
```

Do not print enormous masks by default.

---

### 7. Saved Trajectory

When the user presses save:

```python
record = {
    "timestamp": time.time(),
    "W_T_R": state.W_T_R.tolist(),
    "position": state.W_T_R[:3, 3].tolist(),
    "rpy": matrix_to_rpy(state.W_T_R[:3, :3]),
    "eval": summarize_eval(state.last_eval),
}
state.saved_poses.append(record)
```

Update a trajectory `LineSet` connecting saved pose positions.

Trajectory line color can reflect feasibility:

- simple version: one blue line through saved points.
- better version: use separate small spheres or coordinate frames colored by feasibility.

Export file:

```text
data/teleop_demo_YYYYMMDD_HHMMSS.json
```

JSON structure:

```json
{
  "metadata": {
    "tool": "keyboard_pose_labeler",
    "created_at": "...",
    "translation_step": 0.03,
    "rotation_step_deg": 5.0
  },
  "poses": [
    {
      "timestamp": 0.0,
      "W_T_R": [[...]],
      "position": [...],
      "rpy": [...],
      "eval": {
        "body_valid": true,
        "weak_valid": true,
        "strong_valid": false,
        "leg_counts": [1, 3, 4, 2, 5, 2],
        "score": 2.85
      }
    }
  ]
}
```

---

### 8. Optional Local Direction Hint

Add a pseudo-haptic visual direction hint after the basic tool works.

Around the current pose, evaluate small movements in six local directions:

```text
+X, -X, +Y, -Y, +Z, -Z
```

Compute score differences:

```python
score_delta = score(q + delta) - score(q)
```

Display or print the best improving direction:

```text
hint: move +Y improves score by 2.1
```

Optional visualization: draw a green arrow in the best direction.

This is not required for the first commit.

---

## Loading Data

The first version can assume that environment and robot objects are created the same way as existing examples in `se3_planning.py`.

If initialization is complicated, implement the teleop class independent of initialization and provide a placeholder function:

```python
def build_hex_state_from_existing_project_config():
    raise NotImplementedError("Wire this to existing env_robot_voxels.py setup")
```

Then in the main block:

```python
if __name__ == "__main__":
    hex_state = build_hex_state_from_existing_project_config()
    init_pose = np.eye(4)
    app = KeyboardPoseLabeler(hex_state, init_pose)
    app.run()
```

It is acceptable for the first commit to require the user to edit the init function to match their dataset paths.

---

## Performance Notes

`RobotFeasiCheck` may be expensive. Avoid unnecessary repeated calls.

Recommended:

1. Evaluate only after keyboard pose update.
2. Do not run evaluation every render frame.
3. Cache the last pose/eval if pose unchanged.
4. Keep terminal output compact.
5. For feasible landing visualization, update only when mode is enabled.

---

## Acceptance Criteria

The first version is considered complete if:

1. Running `python keyboard_pose_labeler.py` opens an Open3D window.
2. The environment point cloud is visible.
3. A robot body box is visible.
4. Pressing movement keys moves the body in the same window.
5. Each movement triggers feasibility evaluation.
6. The body color changes green/yellow/red according to strong/weak/invalid.
7. Terminal prints leg counts and score.
8. The user can save poses.
9. The user can export saved poses to JSON.
10. The window updates continuously without recreating itself.

---

## Non-Goals for Version 1

Do not implement these yet:

1. Real robot control.
2. ROS/RViz integration.
3. Force-feedback hardware.
4. SpaceMouse / VR controller.
5. Full robot mesh and all leg kinematics visualization.
6. Learning policy.
7. Automatic path planning.
8. A* local repair.
9. OMPL integration.
10. Multi-threaded UI architecture unless necessary.

---

## Future Extensions

After the first version works:

1. Add SpaceMouse support.
2. Add pseudo-haptic direction arrows.
3. Add top-K pose candidate display around the current pose.
4. Add feasible landing point visualization per leg.
5. Add local point-cloud crop visualization.
6. Add replay mode for saved demonstrations.
7. Add trajectory smoothing.
8. Add dataset export for imitation learning.
9. Add ROS bridge for RViz and real robot state.
10. Add actual haptic device force output.

---

## Design Philosophy

Keep the first version simple.

The purpose is to answer this question:

> Can a human operator, using only keyboard input and visual feasibility feedback, manually guide the robot body through difficult local terrain configurations and produce useful demonstration trajectories?

If yes, later versions can replace keyboard with haptic devices and use the collected trajectories for learning.
