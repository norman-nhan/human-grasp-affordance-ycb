#!/usr/bin/env python3
"""Export point-aligned human-grasp affordances from YCB-Affordance MANO poses.

The source dataset contains posed MANO hand meshes, not object-surface masks.
This script derives reproducible pseudo-contact labels by measuring the distance
from deterministic object-surface samples to deterministic MANO-surface samples.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


SCHEMA = "ycb-human-grasp-affordance/v1"


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load positions and triangulated faces from a Wavefront OBJ."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v" and len(fields) >= 4:
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif fields[0] == "f" and len(fields) >= 4:
                polygon = [int(item.split("/", 1)[0]) for item in fields[1:]]
                polygon = [index - 1 if index > 0 else len(vertices) + index for index in polygon]
                for offset in range(1, len(polygon) - 1):
                    faces.append([polygon[0], polygon[offset], polygon[offset + 1]])
    result_vertices = np.asarray(vertices, dtype=np.float64)
    result_faces = np.asarray(faces, dtype=np.int64)
    if result_vertices.ndim != 2 or result_vertices.shape[1:] != (3,) or not len(result_faces):
        raise ValueError(f"Empty or unsupported OBJ mesh: {path}")
    if not np.isfinite(result_vertices).all():
        raise ValueError(f"Non-finite OBJ vertices: {path}")
    return result_vertices, result_faces


def sample_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted triangle sampling with corresponding unit normals."""
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > 1e-15
    if not valid.any():
        raise ValueError("Mesh has no non-degenerate triangles")
    triangles = triangles[valid]
    cross = cross[valid]
    double_area = double_area[valid]
    selected = rng.choice(len(triangles), size=count, p=double_area / double_area.sum())
    chosen = triangles[selected]
    u = np.sqrt(rng.random(count))
    v = rng.random(count)
    points = ((1.0 - u)[:, None] * chosen[:, 0]
              + (u * (1.0 - v))[:, None] * chosen[:, 1]
              + (u * v)[:, None] * chosen[:, 2])
    normals = cross[selected] / double_area[selected, None]
    return points.astype(np.float32), normals.astype(np.float32)


def object_name_from_grasp(record: dict) -> str:
    body = Path(record["body"])
    if body.name != "nontextured_transformed.wrl" or body.parent.name != "google_16k":
        raise ValueError(f"Unexpected YCB body path: {record['body']!r}")
    return body.parent.parent.name


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def colors_from_score(score: np.ndarray) -> np.ndarray:
    """Blue -> yellow -> red color map without a plotting dependency."""
    value = np.clip(score, 0.0, 1.0)
    red = np.clip(2.0 * value, 0.0, 1.0)
    green = np.clip(2.0 - 2.0 * value, 0.0, 1.0)
    blue = np.clip(1.0 - 2.0 * value, 0.0, 1.0)
    return np.rint(np.column_stack((red, green, blue)) * 255).astype(np.uint8)


def write_ply(
    path: Path,
    points: np.ndarray,
    normals: np.ndarray,
    score: np.ndarray,
    mask: np.ndarray,
    contact_count: np.ndarray,
    min_distance: np.ndarray,
) -> None:
    """Write an ASCII PLY preview while preserving scalar training labels."""
    colors = colors_from_score(score)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        for name in ("x", "y", "z", "nx", "ny", "nz"):
            stream.write(f"property float {name}\n")
        for name in ("red", "green", "blue"):
            stream.write(f"property uchar {name}\n")
        stream.write("property float affordance_score\n")
        stream.write("property uchar affordance_mask\n")
        stream.write("property ushort contact_count\n")
        stream.write("property float min_hand_distance_m\nend_header\n")
        for point, normal, color, soft, hard, hits, distance in zip(
            points, normals, colors, score, mask, contact_count, min_distance
        ):
            values = [*point, *normal, *color, soft, int(hard), int(hits), distance]
            stream.write(
                "{:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} "
                "{} {} {} {:.9g} {} {} {:.9g}\n".format(*values)
            )
    os.replace(temporary, path)


def source_digest(mesh_path: Path, grasp_paths: Iterable[Path]) -> str:
    digest = hashlib.sha256(mesh_path.read_bytes())
    for path in sorted(grasp_paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_grasp(path: Path) -> dict:
    # YCB-Affordance distributes trusted Python pickles. Never use this on an
    # untrusted file because pickle loading can execute arbitrary code.
    with path.open("rb") as stream:
        return pickle.load(stream, encoding="latin")


def resolve_grasp_paths(data_root: Path, specifications: list[str] | None) -> list[Path]:
    """Resolve exact paths or glob expressions in deterministic order."""
    if not specifications:
        paths = list((data_root / "grasps").glob("obj_*"))
    else:
        paths = []
        for specification in specifications:
            matches = [Path(path) for path in glob.glob(specification)]
            if not matches:
                candidate = Path(specification)
                if not candidate.is_absolute():
                    candidate = data_root / "grasps" / candidate
                if candidate.is_file():
                    matches = [candidate]
            if not matches:
                raise FileNotFoundError(
                    f"Grasp path or pattern matched nothing: {specification}"
                )
            paths.extend(matches)
    resolved = sorted({path.resolve() for path in paths})
    if not resolved:
        raise FileNotFoundError("No grasp pickles were selected")
    if any(not path.is_file() for path in resolved):
        raise FileNotFoundError("A selected grasp path is not a file")
    return resolved


def mano_vertices(layer, record: dict, torch_module, device: str) -> np.ndarray:
    rotation = np.asarray(record["pca_manorot"], dtype=np.float32).reshape(1, 3)
    articulation = np.asarray(record["pca_poses"], dtype=np.float32).reshape(1, 45)
    translation = np.asarray(record["mano_trans"], dtype=np.float32).reshape(1, 3)
    pose = torch_module.as_tensor(
        np.concatenate((rotation, articulation), axis=1), device=device
    )
    trans = torch_module.as_tensor(translation, device=device)
    with torch_module.no_grad():
        vertices, _ = layer(pose, th_trans=trans)
    # manopth outputs millimetres; YCB model vertices are metres.
    return vertices[0].detach().cpu().numpy().astype(np.float64) / 1000.0


def derive_labels(
    object_points: np.ndarray,
    hand_surfaces: Iterable[np.ndarray],
    threshold_m: float,
    sigma_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    min_distance = np.full(len(object_points), np.inf, dtype=np.float64)
    contact_count = np.zeros(len(object_points), dtype=np.uint16)
    grasp_count = 0
    for hand_points in hand_surfaces:
        distance = cKDTree(hand_points).query(object_points, workers=-1)[0]
        min_distance = np.minimum(min_distance, distance)
        contact_count += (distance <= threshold_m).astype(np.uint16)
        grasp_count += 1
    if grasp_count == 0:
        raise ValueError("No MANO grasps were supplied")
    mask = min_distance <= threshold_m
    frequency = contact_count.astype(np.float32) / float(grasp_count)
    proximity = np.exp(-0.5 * np.square(min_distance / sigma_m)).astype(np.float32)
    return min_distance.astype(np.float32), contact_count, frequency, proximity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/human_grasp_affordance")
    )
    parser.add_argument("--objects", nargs="*", help="Optional YCB directory names")
    parser.add_argument(
        "--grasp-paths",
        nargs="+",
        help=(
            "Exact grasp pickle paths or quoted glob patterns. Basenames are "
            "resolved under DATA_ROOT/grasps. By default every grasp is used."
        ),
    )
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--hand-surface-points", type=int, default=12000)
    parser.add_argument("--contact-threshold-mm", type=float, default=5.0)
    parser.add_argument("--sigma-mm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-objects", type=int)
    parser.add_argument("--max-grasps-per-object", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.points < 1 or args.hand_surface_points < 1:
        parser.error("point counts must be positive")
    if args.contact_threshold_mm <= 0 or args.sigma_mm <= 0:
        parser.error("distance settings must be positive")
    return args


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_dir.resolve()
    grasp_paths = resolve_grasp_paths(data_root, args.grasp_paths)

    grouped: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for path in grasp_paths:
        record = load_grasp(path)
        grouped[object_name_from_grasp(record)].append((path, record))
    selected = sorted(grouped)
    if args.objects:
        unknown = sorted(set(args.objects) - set(selected))
        if unknown:
            raise ValueError(f"Objects without annotations: {unknown}")
        selected = [name for name in selected if name in set(args.objects)]
    if args.max_objects is not None:
        selected = selected[: args.max_objects]

    try:
        import torch
        from manopth.manolayer import ManoLayer
    except ImportError as error:
        raise RuntimeError(
            "This exporter needs the same torch/manopth environment used by "
            "visualize_grasps.py"
        ) from error

    layer = ManoLayer(
        mano_root=str(data_root / "mano"),
        side="right",
        use_pca=True,
        ncomps=45,
        flat_hand_mean=True,
    ).to(args.device)
    hand_faces = layer.th_faces.detach().cpu().numpy().astype(np.int64)
    threshold_m = args.contact_threshold_mm / 1000.0
    sigma_m = args.sigma_mm / 1000.0
    manifest = {
        "schema": SCHEMA,
        "source_data_root": str(data_root),
        "settings": vars(args) | {
            "data_root": str(data_root),
            "output_dir": str(output_root),
            "grasp_paths": [str(path) for path in grasp_paths],
        },
        "objects": [],
    }

    for object_index, name in enumerate(selected):
        target = output_root / name
        npz_path = target / "affordance.npz"
        metadata_path = target / "metadata.json"
        ply_path = target / "affordance.ply"
        if npz_path.exists() and metadata_path.exists() and not args.overwrite:
            existing = json.loads(metadata_path.read_text())
            manifest["objects"].append(
                {
                    "object_id": name,
                    "status": "completed_existing",
                    "num_grasps": existing["num_grasps"],
                    "positive_points": existing["positive_points"],
                    "positive_fraction": existing["positive_fraction"],
                    "output": str(npz_path),
                }
            )
            print(f"[{object_index + 1}/{len(selected)}] {name}: existing output skipped", flush=True)
            continue

        records = grouped[name]
        if args.max_grasps_per_object is not None:
            records = records[: args.max_grasps_per_object]
        mesh_path = data_root / "models" / name / "google_16k" / "textured.obj"
        vertices, faces = load_obj(mesh_path)
        extent = np.ptp(vertices, axis=0)
        if not 0.001 < float(np.linalg.norm(extent)) < 2.0:
            raise ValueError(f"Unexpected object scale for {name}: extent={extent.tolist()}")
        object_rng = np.random.default_rng(args.seed)
        object_points, object_normals = sample_surface(vertices, faces, args.points, object_rng)

        hand_surfaces = []
        taxonomy = Counter()
        for grasp_index, (_, record) in enumerate(records):
            hand_vertices = mano_vertices(layer, record, torch, args.device)
            hand_rng = np.random.default_rng(args.seed + grasp_index + 1)
            hand_points, _ = sample_surface(
                hand_vertices, hand_faces, args.hand_surface_points, hand_rng
            )
            hand_surfaces.append(hand_points)
            taxonomy[int(record["taxonomy"])] += 1

        min_distance, contact_count, frequency, proximity = derive_labels(
            object_points, hand_surfaces, threshold_m, sigma_m
        )
        mask = min_distance <= threshold_m
        atomic_npz(
            npz_path,
            points=object_points,
            normals=object_normals,
            affordance_score=proximity,
            contact_frequency=frequency,
            affordance_mask=mask,
            contact_count=contact_count,
            min_hand_distance_m=min_distance,
        )
        write_ply(
            ply_path,
            object_points,
            object_normals,
            proximity,
            mask,
            contact_count,
            min_distance,
        )
        metadata = {
            "schema": SCHEMA,
            "object_id": name,
            "units": "metres",
            "source_mesh": str(mesh_path),
            "source_grasps": [path.name for path, _ in records],
            "source_sha256": source_digest(mesh_path, (path for path, _ in records)),
            "num_object_points": len(object_points),
            "num_grasps": len(records),
            "taxonomy_histogram": {str(key): value for key, value in sorted(taxonomy.items())},
            "contact_threshold_m": threshold_m,
            "proximity_sigma_m": sigma_m,
            "hand_surface_points_per_grasp": args.hand_surface_points,
            "seed": args.seed,
            "positive_points": int(mask.sum()),
            "positive_fraction": float(mask.mean()),
            "minimum_distance_m": float(min_distance.min()),
            "maximum_affordance_score": float(proximity.max()),
            "label_definition": (
                "Pseudo-contact derived from distance to annotated MANO hand surfaces; "
                "not measured pressure, force, or a source-provided point mask."
            ),
            "arrays": {
                "points": "float32 [N,3]",
                "normals": "float32 [N,3]",
                "affordance_score": "float32 [N], exp(-0.5*(minimum_distance/sigma)^2)",
                "contact_frequency": "float32 [N], fraction of grasps within threshold",
                "affordance_mask": "bool [N], minimum_distance <= threshold",
                "contact_count": "uint16 [N]",
                "min_hand_distance_m": "float32 [N]",
            },
        }
        atomic_json(metadata_path, metadata)
        manifest["objects"].append(
            {
                "object_id": name,
                "status": "completed",
                "num_grasps": len(records),
                "positive_points": int(mask.sum()),
                "positive_fraction": float(mask.mean()),
                "output": str(npz_path),
            }
        )
        print(
            f"[{object_index + 1}/{len(selected)}] {name}: "
            f"{len(records)} grasps, {int(mask.sum())}/{len(mask)} positive points",
            flush=True,
        )

    manifest["summary"] = {
        "selected_objects": len(selected),
        "completed": sum(
            item["status"] in ("completed", "completed_existing")
            for item in manifest["objects"]
        ),
        "reused_existing": sum(
            item["status"] == "completed_existing" for item in manifest["objects"]
        ),
    }
    atomic_json(output_root / "manifest.json", manifest)
    print(f"Manifest: {output_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
