"""Benchmark existing body surface sampling without rebuilding or changing caches.

Run from the repository root:
    python3 scripts/benchmark_body_surface_sampling.py

Extracts the actual Voxels and body-query implementation with AST to avoid
initializing the simulator, rebuilding caches, or allocating full grid centers.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from spatialmath import SE3
import trimesh


ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "legged_gym/expert_complex_utils"
OUTPUT = ROOT / "data/body_surface_sampling_benchmark"
COUNTS = (256, 512, 1024, 2048)
TRIALS = 20


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = ast.parse((UTILS / "env_robot_voxels.py").read_text())
    classes = {n.name: n for n in source.body if isinstance(n, ast.ClassDef)}
    hex_nodes = [
        n for n in classes["HexState"].body
        if (isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id.startswith("BODY_") for t in n.targets
        )) or (isinstance(n, ast.FunctionDef) and n.name == "_BodyEnvironmentDistances")
    ]
    shell = ast.ClassDef(name="BodyQuery", bases=[], keywords=[], body=hex_nodes, decorator_list=[])
    module = ast.Module(body=[
        ast.parse("from __future__ import annotations").body[0],
        classes["Voxels"], shell,
    ], type_ignores=[])
    namespace = dict(np=np, map_coordinates=map_coordinates)
    exec(compile(ast.fix_missing_locations(module), "<actual body query>", "exec"), namespace)
    with np.load(ROOT / "resources/environments/structure2/complex_surface4_point_map.npz") as data:
        bounds = data["bounds"]
    with np.load(UTILS / "voxels_info/env_voxels.npz") as data:
        field = data["esdf"]
    voxel = namespace["Voxels"].__new__(namespace["Voxels"])
    voxel._bounds = bounds
    voxel.voxel_scale = 0.01
    voxel.grid_shape = voxel._compute_grid_shape()
    assert tuple(voxel.grid_shape) == field.shape, "ESDF cache does not match assumed 1 cm grid"
    query = namespace["BodyQuery"]()
    body = SimpleNamespace()
    query.robot_voxels = SimpleNamespace(body_voxels=body)
    query.env_pointsmap_voxels = SimpleNamespace(voxels=voxel, env_esdf=field)
    mesh = trimesh.load_mesh(ROOT / "resources/robots/hex_magnetic/body_env_collision.STL")
    path_file = UTILS / "SE3_path/optimized_cma_es_torch_path_20261104_110353_995896.json"
    path = json.loads(path_file.read_text())["dense_poses"]
    rows = np.linspace(0, len(path["t"]) - 1, 32).astype(int)
    base_poses = [SE3(path["t"][i]) * SE3.AngleAxis(path["ang"][i], path["vec"][i]) for i in rows]
    rng = np.random.default_rng(77)
    poses = list(base_poses)
    for sigma in (0.02, 0.05):
        poses.extend(SE3(p.t + rng.normal(0, sigma, 3)) * SE3.RPY(
            rng.normal(0, np.deg2rad(3), 3)
        ) * SE3.Rt(p.R) for p in base_poses)

    def evaluate(points, pose):
        body.bounding_points = points
        _, distances = query._BodyEnvironmentDistances(pose)
        violation = np.maximum(0, (query.BODY_SOFT_CLEARANCE - distances) / query.BODY_SOFT_CLEARANCE)
        return float(query.BODY_COST_WEIGHT * np.mean(violation ** 2)), float(distances.min())

    print("Loaded map", field.shape, "poses", len(poses), flush=True)
    reference_points, _ = trimesh.sample.sample_surface(mesh, 262144, seed=901)
    reference = np.array([evaluate(reference_points, p) for p in poses])
    half_reference = np.array([evaluate(reference_points[:131072], p) for p in poses])
    assert np.isfinite(reference).all(), "Benchmark includes invalid map queries"
    unsafe = reference[:, 1] < query.BODY_HARD_CLEARANCE
    active = reference[:, 0] > 1e-8
    print("Dense reference complete; unsafe poses", unsafe.sum(), "active costs", active.sum(), flush=True)
    results = []
    raw = {}
    # The same 2048 samples supply nested subsets; each trial uses a new seed.
    samples = [trimesh.sample.sample_surface(mesh, max(COUNTS), seed=1000 + i)[0] for i in range(TRIALS)]
    for count in COUNTS:
        trials = np.array([[evaluate(points[:count], p) for p in poses] for points in samples])
        raw[str(count)] = trials.tolist()
        cost_error = np.abs(trials[:, :, 0] - reference[None, :, 0])
        optimistic = np.maximum(trials[:, :, 1] - reference[None, :, 1], 0) * 1000
        # Randomized, repeated blocks time warm queries across different poses.
        body.bounding_points = samples[0][:count]
        for p in poses[:8]:
            query._BodyEnvironmentDistances(p)
        timing = {"interpolation": [], "body_query": [], "body_query_and_reductions": []}
        world = [(p * body.bounding_points.T).T for p in poses]
        for _ in range(9):
            for name in rng.permutation(list(timing)):
                t = perf_counter()
                for _ in range(3):
                    for i, p in enumerate(poses):
                        if name == "interpolation":
                            voxel.TrilinearSample(field, world[i], -query.BODY_SOFT_CLEARANCE)
                        elif name == "body_query":
                            query._BodyEnvironmentDistances(p)
                        else:
                            _, d = query._BodyEnvironmentDistances(p)
                            np.all(d >= query.BODY_HARD_CLEARANCE)
                            query.BODY_COST_WEIGHT * np.mean(np.maximum(
                                0, (query.BODY_SOFT_CLEARANCE - d) / query.BODY_SOFT_CLEARANCE
                            ) ** 2)
                timing[name].append((perf_counter() - t) * 1e6 / (3 * len(poses)))
        # Coverage estimates on independent dense surface probes; not an upper bound.
        coverages = [cKDTree(points[:count]).query(reference_points[:32768])[0] for points in samples]
        item = dict(
            count=count,
            characteristic_spacing_mm=float(np.sqrt(mesh.area / count) * 1000),
            timing_us={k: dict(median=float(np.median(v)), p95=float(np.percentile(v, 95))) for k, v in timing.items()},
            active_cost_mae=float(cost_error[:, active].mean()),
            active_cost_p95_absolute_error=float(np.percentile(cost_error[:, active], 95)),
            active_cost_normalized_mae=float(cost_error[:, active].mean() / reference[active, 0].mean()),
            false_safe_count=int(np.count_nonzero(trials[:, unsafe, 1] >= query.BODY_HARD_CLEARANCE)),
            unsafe_trial_count=int(TRIALS * unsafe.sum()),
            false_safe_rate=float(np.mean(trials[:, unsafe, 1] >= query.BODY_HARD_CLEARANCE)),
            optimistic_min_clearance_p95_mm=float(np.percentile(optimistic, 95)),
            optimistic_min_clearance_max_mm=float(optimistic.max()),
            coverage_p95_mm=float(np.percentile(coverages, 95) * 1000),
            coverage_observed_max_mm=float(np.max(coverages) * 1000),
        )
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    report = dict(
        method="Existing area-weighted random surface samples, fixed per trial; actual body query via AST extraction",
        path=str(path_file.relative_to(ROOT)),
        pose_design="32 evenly selected path poses, plus 32 each with translation sigma 2/5 cm and rotation sigma 3 degrees",
        pose_count=len(poses), trial_count=TRIALS, mesh_area_m2=float(mesh.area),
        map_shape=list(field.shape), voxel_scale_m=voxel.voxel_scale,
        hard_clearance_m=query.BODY_HARD_CLEARANCE, soft_clearance_m=query.BODY_SOFT_CLEARANCE,
        body_cost_weight=query.BODY_COST_WEIGHT,
        reference_count=len(reference_points), reference_unsafe_count=int(unsafe.sum()),
        reference_active_cost_count=int(active.sum()),
        reference_half_vs_full_cost_mae=float(np.abs(reference[:, 0] - half_reference[:, 0]).mean()),
        reference_half_vs_full_hard_disagreements=int(np.count_nonzero(
            (reference[:, 1] >= query.BODY_HARD_CLEARANCE) != (half_reference[:, 1] >= query.BODY_HARD_CLEARANCE)
        )),
        notes=["Dense samples are a reference, not geometric ground truth.",
               "Timing excludes sampling, loading, and all leg computations; p95 is across block averages.",
               "ESDF cache lacks provenance metadata; its dimensions match current map bounds at 1 cm."],
        results=results,
    )
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (OUTPUT / "raw.json").write_text(json.dumps(dict(reference=reference.tolist(), trials=raw), ensure_ascii=False) + "\n")
    print("Saved", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
