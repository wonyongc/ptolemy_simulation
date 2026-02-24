"""ptolemy command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ptolemy_simulation.adapters.registry import build_default_registry
from ptolemy_simulation.analysis.pitch import aggregate_pitch_files, plot_pitch
from ptolemy_simulation.analysis.transmission import (
    aggregate_all,
    aggregate_theta,
    plot_aggregate,
    scan_theta_phi_files,
)
from ptolemy_simulation.config.loader import load_study_bundle
from ptolemy_simulation.pipeline.compile import compile_variants
from ptolemy_simulation.pipeline.matrix import expand_study
from ptolemy_simulation.pipeline.run import run_study
from ptolemy_simulation.postprocess.trajectory import (
    derive_output_path,
    process_cst_traj,
    process_cst_traj_no_fields,
    save_gcs_traj,
)


def _cmd_validate(args: argparse.Namespace) -> int:
    bundle = load_study_bundle(args.study)
    variants = expand_study(bundle)
    result = {
        "study": str(bundle.study.path),
        "study_name": bundle.study.raw["study_name"],
        "simulator": bundle.simulator.raw["simulator"],
        "run_mode": bundle.run.raw["mode"],
        "run_count": len(variants),
    }
    print(json.dumps(result, indent=2))
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    bundle = load_study_bundle(args.study)
    variants = expand_study(bundle)
    registry = build_default_registry()

    study_name = bundle.study.raw["study_name"]
    output_root = Path(str(bundle.study.raw.get("output_root", "generated/studies")))

    artifacts = compile_variants(registry, study_name=study_name, output_root=output_root, variants=variants)
    print(
        json.dumps(
            {
                "study_name": study_name,
                "run_count": len(variants),
                "artifacts": [
                    {"run_id": artifact.run_id, "files": {k: str(v) for k, v in artifact.files.items()}}
                    for artifact in artifacts
                ],
            },
            indent=2,
        )
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    result = run_study(args.study, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0


def _cmd_analyze_transmission(args: argparse.Namespace) -> int:
    mapping = scan_theta_phi_files(args.input_dir)
    theta_data = aggregate_theta(mapping, particles_per_run=args.particles_per_run)

    if args.output_plot:
        plot_aggregate(theta_data, args.output_plot)

    z, total_counts, avg_energy = aggregate_all(mapping)
    summary = {
        "theta_count": len(theta_data),
        "files": sum(len(v) for v in mapping.values()),
        "global_points": int(len(z)),
        "last_count": float(total_counts[-1]) if len(total_counts) else None,
        "last_energy": float(avg_energy[-1]) if len(avg_energy) else None,
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_analyze_pitch(args: argparse.Namespace) -> int:
    if args.input_files:
        files = args.input_files
    else:
        files = sorted(args.input_dir.glob("*.txt"))
    z, avg_pitch = aggregate_pitch_files(files, x_window_abs=args.x_window_abs)
    if args.output_plot:
        plot_pitch(z, avg_pitch, args.output_plot)
    summary = {
        "files": len(files),
        "points": int(len(z)),
        "avg_pitch_mean": float(avg_pitch.mean()) if len(avg_pitch) else None,
    }
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_traj_convert(args: argparse.Namespace) -> int:
    for input_path in args.inputs:
        if not input_path.exists():
            raise SystemExit(f"Input not found: {input_path}")

        output_npz = derive_output_path(input_path, args.output_dir, ".npz")
        if args.mode == "traj":
            process_cst_traj(input_path, output_npz)
        else:
            process_cst_traj_no_fields(input_path, output_npz)
        print(f"Wrote {output_npz}")

        if args.gcs:
            output_gcs = derive_output_path(output_npz, args.output_dir, "_gcs.npz")
            save_gcs_traj(output_npz, output_gcs, prefix=args.prefix)
            print(f"Wrote {output_gcs}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ptolemy", description="PTOLEMY transmission pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Validate study and referenced config files")
    validate.add_argument("--study", type=Path, required=True)
    validate.set_defaults(func=_cmd_validate)

    compile_cmd = sub.add_parser("compile", help="Compile all study runs into simulator artifacts")
    compile_cmd.add_argument("--study", type=Path, required=True)
    compile_cmd.set_defaults(func=_cmd_compile)

    run_cmd = sub.add_parser("run", help="One-click compile + stage + submit pipeline")
    run_cmd.add_argument("--study", type=Path, required=True)
    run_cmd.add_argument("--dry-run", action="store_true", help="Skip remote command execution")
    run_cmd.set_defaults(func=_cmd_run)

    analyze = sub.add_parser("analyze", help="Analysis helpers")
    analyze_sub = analyze.add_subparsers(dest="analyze_command", required=True)

    analyze_trans = analyze_sub.add_parser("transmission", help="Aggregate transmission monitor exports")
    analyze_trans.add_argument("--input-dir", type=Path, required=True)
    analyze_trans.add_argument("--particles-per-run", type=int, default=10200)
    analyze_trans.add_argument("--output-plot", type=Path, default=None)
    analyze_trans.add_argument("--summary-json", type=Path, default=None)
    analyze_trans.set_defaults(func=_cmd_analyze_transmission)

    analyze_pitch = analyze_sub.add_parser("pitch", help="Aggregate pitch monitor exports")
    analyze_pitch.add_argument("--input-dir", type=Path, default=Path("."))
    analyze_pitch.add_argument("--input-files", type=Path, nargs="*", default=None)
    analyze_pitch.add_argument("--x-window-abs", type=float, default=0.005)
    analyze_pitch.add_argument("--output-plot", type=Path, default=None)
    analyze_pitch.set_defaults(func=_cmd_analyze_pitch)

    traj = sub.add_parser("traj", help="Trajectory conversion helpers")
    traj_sub = traj.add_subparsers(dest="traj_command", required=True)
    traj_convert = traj_sub.add_parser("convert", help="Convert CST CSV trajectories to NPZ/GCS")
    traj_convert.add_argument("inputs", type=Path, nargs="+")
    traj_convert.add_argument("--mode", choices=["traj", "traj_nofields"], default="traj")
    traj_convert.add_argument("--gcs", action="store_true")
    traj_convert.add_argument("--prefix", type=int, default=3)
    traj_convert.add_argument("--output-dir", type=Path, default=None)
    traj_convert.set_defaults(func=_cmd_traj_convert)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
