# PTOLEMY Simulation Pipeline

Adapter-first simulation pipeline for the PTOLEMY tritium beta-decay detector.
Supports parameter sweeps, CST Studio macro generation, and post-run analysis.

## What You Need
- Python 3.9+
- CST Studio installed (for CST studies)
- This repo cloned locally

## One-Time Setup
From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Check the CLI is available:

```bash
ptolemy --help
```

## Quick Start -- Local CST Study

```bash
ptolemy run --study config/examples/study.minimal.local.yaml
```

What this does:
1. Validates study, detector, simulator, and run configs
2. Expands the parameter matrix (if any) into run variants
3. Compiles CST macros via Jinja2 templates
4. Stages project files into your local CST run folder

For the default local config, staged files go under:
- `/mnt/c/Users/<user>/ptolemy/ptx/runs/<study_name>/`

## Run Inside CST
Open the staged `.cst` file in CST, then run macro:
- `Model/3D/PTOLEMY_run.mcr`

This wrapper macro does:
1. Build geometry (`PTOLEMY_build.mcs`)
2. Run solver
3. Export outputs

## Config Structure

A study references three config files:

| Config | Purpose | Examples |
|--------|---------|----------|
| **Detector** | Geometry, sources, field profiles | `detector.minimal.yaml`, `detector.ptolemy_full.yaml` |
| **Simulator** | Solver settings, mesh, monitors, export toggles | `sim.cst.minimal.yaml`, `sim.cst.ptolemy.yaml` |
| **Run** | Execution mode (local or Della), paths, SLURM settings | `run.local.yaml`, `run.della.yaml` |

All example configs live in `config/examples/`.

### Parameter Sweeps (Matrix)

Studies can sweep over parameters using a Cartesian product matrix.
Override values are applied via JSON pointers (RFC 6901):

```yaml
# config/examples/study.theta_phi.yaml
matrix:
  theta_deg: [10, 20, 30, 40, 50, 60, 70, 80]
  phi_deg: [0, 45, 90, 135, 180, 225, 270, 315]
overrides:
  detector:
    "/sources/singles/0/pitches/0": "$theta_deg"
    "/sources/singles/0/phis/0": "$phi_deg"
run_name_template: "th{theta_deg}_phi{phi_deg}"
```

This generates 64 run variants (8 theta x 8 phi).

### Export Toggles

In the simulator config (`sim.*.yaml`), under `toggles`:

- `export_transmission` -- transmission monitor data
- `export_pitch_analysis` -- pitch angle data
- `export_trajectories` -- particle trajectory CSV
- `export_field_maps` -- E/B/potential field maps

Field map regions are configured under `postprocess.field_exports.regions`.

## Where Outputs Go
For local runs, outputs are written per variant to the `results/` subdirectory:
- `results/transmission.txt`
- `results/pitchanalysis.txt`
- `results/trajectories.csv`
- `results/fields/<run>_<region>_B.txt`
- `results/fields/<run>_<region>_E.txt`
- `results/fields/<run>_<region>_phi.txt`

## CLI Reference

### Core pipeline

```bash
# Validate only
ptolemy validate --study config/examples/study.minimal.local.yaml

# Compile only (generate macros, no staging)
ptolemy compile --study config/examples/study.minimal.local.yaml

# Full pipeline: compile + stage (+ submit on Della)
ptolemy run --study config/examples/study.minimal.local.yaml
```

### Analysis

```bash
# Aggregate transmission monitor exports
ptolemy analyze transmission --input-dir <DIR> \
    [--particles-per-run 10200] \
    [--output-plot plot.png] \
    [--summary-json summary.json]

# Aggregate pitch angle data
ptolemy analyze pitch --input-dir <DIR> \
    [--input-files FILE ...] \
    [--x-window-abs 0.005] \
    [--output-plot plot.png]
```

### Trajectory conversion

```bash
# Convert CST trajectory CSV to NPZ
ptolemy traj convert INPUT1 [INPUT2 ...] \
    [--mode traj|traj_nofields] \
    [--output-dir DIR]

# Also extract GCS (guiding-centre) segments
ptolemy traj convert INPUT1 --gcs [--prefix 3]
```

## Example Studies

| Study | Description |
|-------|-------------|
| `study.minimal.local.yaml` | Minimal single-run local CST study |
| `study.single.local.yaml` | Single-run local study with full detector |
| `study.ptolemy.local.yaml` | Full PTOLEMY detector, local staging |
| `study.ptolemy.della.yaml` | Full PTOLEMY detector, Della cluster |
| `study.theta_phi.yaml` | Theta/phi sweep (64 variants) on Della |
| `study.kass.stub.yaml` | Kassiopeia adapter stub |

## Della Dry Run (Generate Scripts, Don't Submit)
```bash
ptolemy run --study config/examples/study.ptolemy.della.yaml --dry-run
```

## Supported Simulators

- **CST Studio** -- fully implemented adapter with Jinja2 macro generation
- **Kassiopeia** -- stub adapter (placeholder, not yet implemented)

## Project Layout

```
ptolemy_simulation/
  adapters/        # Simulator adapters (CST, Kassiopeia)
  analysis/        # Transmission and pitch angle analysis
  config/          # Config loading, validation, models
  data_models/     # Trajectory data model
  pipeline/        # Matrix expansion, compilation, staging
  postprocess/     # Trajectory conversion, monitor parsing
  templates/cst/   # Jinja2 CST macro templates
  utils/           # Hashing, JSON pointers, naming, templating
  cli.py           # CLI entry point
config/examples/   # Example YAML configs
tests/             # Test suite
```
