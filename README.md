# PTOLEMY Transmission: Quick Run Guide

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

## Fastest Way to Run a Local CST Study
Use the minimal local example:

```bash
ptolemy run --study config/examples/study.minimal.local.yaml
```

What this does:
1. Validates config
2. Compiles CST macros
3. Stages project files into your local CST run folder

For the default local config in this repo, staged files go under:
- `/mnt/c/Users/wonyongc/ptolemy/ptx/runs/<study_name>/`

## Run Inside CST
Open the staged `.cst` file in CST, then run macro:
- `Model/3D/PTOLEMY_run.mcr`

This wrapper macro does:
1. Build geometry (`PTOLEMY_build.mcs`)
2. Run solver
3. Export outputs

## Where Outputs Go
For the minimal local setup, outputs are written to:
- `.../run-0000/results/transmission.txt`
- `.../run-0000/results/pitchanalysis.txt`
- `.../run-0000/results/trajectories.csv`
- `.../run-0000/results/run-0000_trajectories_raw_<n>.txt`
- `.../run-0000/results/fields/run-0000_<region>_B.txt`
- `.../run-0000/results/fields/run-0000_<region>_E.txt`
- `.../run-0000/results/fields/run-0000_<region>_phi.txt`

Field files include header metadata for:
- field mode
- x/y/z limits
- step size
- column names

## Change What Gets Exported
Edit:
- `config/examples/sim.cst.minimal.yaml`

Main toggles:
- `toggles.export_transmission`
- `toggles.export_pitch_analysis`
- `toggles.export_trajectories`
- `toggles.export_field_maps`

Field map region settings:
- `postprocess.field_exports.regions`
- each region defines `x/y/z` start, stop, step

## Run the Full PTOLEMY Local Study
```bash
ptolemy run --study config/examples/study.ptolemy.local.yaml
```

## Della Dry Run (Generate Scripts, Don’t Submit)
```bash
ptolemy run --study config/examples/study.ptolemy.della.yaml --dry-run
```

## Minimal Command Reference
Validate only:
```bash
ptolemy validate --study config/examples/study.minimal.local.yaml
```

Compile only:
```bash
ptolemy compile --study config/examples/study.minimal.local.yaml
```

Full pipeline (compile + stage):
```bash
ptolemy run --study config/examples/study.minimal.local.yaml
```
