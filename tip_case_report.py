#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from geometry_io import build_geometry_snapshot, parse_geometry
from rcs_solver import solve_monostatic_rcs_2d


def _parse_csv_floats(text: str) -> List[float]:
    out: List[float] = []
    for token in (text or "").split(","):
        t = token.strip()
        if not t:
            continue
        out.append(float(t))
    return out


def _group_samples(samples: List[Dict[str, Any]]) -> Dict[Tuple[float, float], Dict[str, Any]]:
    by_key: Dict[Tuple[float, float], Dict[str, Any]] = {}
    for row in samples:
        key = (float(row.get("frequency_ghz", 0.0)), float(row.get("theta_inc_deg", 0.0)))
        by_key[key] = row
    return by_key


def _run_case(path: Path, freqs: List[float], elevs: List[float], pol: str, units: str) -> Dict[str, Any]:
    text = path.read_text()
    title, segments, ibcs, dielectrics = parse_geometry(text)
    snapshot = build_geometry_snapshot(title, segments, ibcs, dielectrics)
    result = solve_monostatic_rcs_2d(
        geometry_snapshot=snapshot,
        frequencies_ghz=freqs,
        elevations_deg=elevs,
        polarization=pol,
        geometry_units=units,
        material_base_dir=str(path.parent),
        parallel_elevations=False,
    )
    metadata = result.get("metadata", {}) or {}
    samples = result.get("samples", []) or []
    worst = max(samples, key=lambda r: float(r.get("linear_residual", 0.0))) if samples else None
    return {
        "file": str(path),
        "metadata": {
            "panel_count": metadata.get("panel_count"),
            "residual_norm_max": metadata.get("residual_norm_max"),
            "residual_norm_mean": metadata.get("residual_norm_mean"),
            "junction_nodes": metadata.get("junction_nodes"),
            "junction_constraints": metadata.get("junction_constraints"),
            "junction_trace_constraints": metadata.get("junction_trace_constraints"),
            "junction_flux_constraints": metadata.get("junction_flux_constraints"),
            "junction_orientation_conflict_nodes": metadata.get("junction_orientation_conflict_nodes"),
            "warnings": metadata.get("warnings", []),
        },
        "worst_residual_sample": worst,
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run side-by-side report for baseline vs mixed-tip .geo cases."
    )
    parser.add_argument(
        "--baseline",
        default="airfoil_symmetric.geo",
        help="Baseline .geo path (default: airfoil_symmetric.geo)",
    )
    parser.add_argument(
        "--test",
        default="airfoil_tip_mixed_213_debug.geo",
        help="Test .geo path (default: airfoil_tip_mixed_213_debug.geo)",
    )
    parser.add_argument(
        "--freqs",
        default="4.0",
        help='Comma-separated GHz list (default: "4.0")',
    )
    parser.add_argument(
        "--elevs",
        default="0,15,30,45,60,75,90,105,120,135,150,165,180",
        help='Comma-separated incidence angles in degrees (default: 0..180 by 15)',
    )
    parser.add_argument("--pol", default="TM", choices=["TM", "TE"], help="Polarization")
    parser.add_argument("--units", default="inches", choices=["inches", "meters"], help="Geometry units")
    args = parser.parse_args()

    freqs = _parse_csv_floats(args.freqs)
    elevs = _parse_csv_floats(args.elevs)
    if not freqs:
        raise ValueError("No frequencies provided.")
    if not elevs:
        raise ValueError("No elevations provided.")

    baseline_path = Path(args.baseline).resolve()
    test_path = Path(args.test).resolve()

    baseline = _run_case(baseline_path, freqs, elevs, args.pol, args.units)
    test = _run_case(test_path, freqs, elevs, args.pol, args.units)

    by_b = _group_samples(baseline["samples"])
    by_t = _group_samples(test["samples"])

    keys = sorted(set(by_b.keys()) & set(by_t.keys()))
    deltas: List[Dict[str, Any]] = []
    for key in keys:
        b = by_b[key]
        t = by_t[key]
        delta_db = float(t.get("rcs_db", 0.0)) - float(b.get("rcs_db", 0.0))
        deltas.append(
            {
                "frequency_ghz": key[0],
                "theta_inc_deg": key[1],
                "delta_rcs_db_test_minus_baseline": delta_db,
                "delta_abs_db": abs(delta_db),
            }
        )
    worst_delta = max(deltas, key=lambda d: float(d.get("delta_abs_db", 0.0))) if deltas else None

    report = {
        "inputs": {
            "baseline": str(baseline_path),
            "test": str(test_path),
            "freqs_ghz": freqs,
            "elevs_deg": elevs,
            "polarization": args.pol,
            "units": args.units,
        },
        "baseline": {
            "file": baseline["file"],
            "metadata": baseline["metadata"],
            "worst_residual_sample": baseline["worst_residual_sample"],
        },
        "test": {
            "file": test["file"],
            "metadata": test["metadata"],
            "worst_residual_sample": test["worst_residual_sample"],
        },
        "comparison": {
            "common_sample_count": len(deltas),
            "worst_delta_sample": worst_delta,
            "deltas": deltas,
        },
    }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
