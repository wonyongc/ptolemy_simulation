from __future__ import annotations

from pathlib import Path

import numpy as np

from ptolemy_simulation.analysis.pitch import aggregate_pitch_files
from ptolemy_simulation.analysis.transmission import aggregate_theta, scan_theta_phi_files
from ptolemy_simulation.postprocess.trajectory import (
    derive_output_path,
    process_cst_traj,
    process_cst_traj_no_fields,
    save_gcs_traj,
)


def test_traj_convert_and_gcs(tmp_path: Path) -> None:
    csv = tmp_path / "traj.csv"
    rows = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [1, 0.1, 0.2, 0.3, 1, 0.1, -0.1, 0, 0, 0, 0, 1, 0, 0],
            [2, 0.2, 0.1, 0.2, 1, -0.1, 0.1, 0, 0, 0, 0, 1, 0, 0],
            [3, 0.3, 0.0, 0.1, 1, 0.2, -0.2, 0, 0, 0, 0, 1, 0, 0],
            [4, 0.4, -0.1, 0.0, 1, -0.2, 0.2, 0, 0, 0, 0, 1, 0, 0],
        ]
    )
    np.savetxt(csv, rows, delimiter=",")

    npz = derive_output_path(csv, tmp_path, ".npz")
    process_cst_traj(csv, npz)
    assert npz.exists()

    gcs = derive_output_path(npz, tmp_path, "_gcs.npz")
    save_gcs_traj(npz, gcs, prefix=1)
    assert gcs.exists()


def test_traj_convert_no_fields(tmp_path: Path) -> None:
    csv = tmp_path / "traj_nf.csv"
    rows = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0.1, 0.2, 0.3, 1, 0.1, -0.1, 0],
        ]
    )
    np.savetxt(csv, rows, delimiter=",")

    npz = derive_output_path(csv, tmp_path, ".npz")
    process_cst_traj_no_fields(csv, npz)
    assert npz.exists()


def test_transmission_and_pitch_aggregation(tmp_path: Path) -> None:
    trans1 = tmp_path / "th10_phi0.txt"
    trans2 = tmp_path / "th10_phi45.txt"
    np.savetxt(trans1, np.array([[0.0, 100, 18000], [1.0, 80, 17900]]), delimiter=",")
    np.savetxt(trans2, np.array([[0.0, 110, 18100], [1.0, 70, 17850]]), delimiter=",")

    mapping = scan_theta_phi_files(tmp_path)
    theta_data = aggregate_theta(mapping, particles_per_run=100)
    assert 10 in theta_data
    assert theta_data[10]["count_percent_sum"].shape[0] == 2

    pitch_file = tmp_path / "pitch.txt"
    np.savetxt(
        pitch_file,
        np.array(
            [
                [0.0, 1, 0.001, 0, 1, 0, 0, 18000],
                [0.5, 2, 0.002, 0, 1, 0, 0, 18000],
            ]
        ),
        delimiter=",")

    z, avg = aggregate_pitch_files([pitch_file])
    assert len(z) == len(avg) == 2
