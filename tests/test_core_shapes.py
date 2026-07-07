import numpy as np

from mirnov_rmt.geometry import toroidal_angles
from mirnov_rmt.mode_fit import estimate_n
from mirnov_rmt.rmt import eigen_decomp, mp_edges
from mirnov_rmt.snapshots import fourier_snapshots
from mirnov_rmt.spectral import coherence_matrix, cross_spectral_matrix
from mirnov_rmt.synthetic import generate_mirnov


def test_toroidal_angles_shape():
    phi = toroidal_angles(8)
    assert phi.shape == (8,)
    assert np.isclose(phi[0], 0.0)


def test_coherence_diagonal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 10)) + 1j * rng.normal(size=(4, 10))
    C = coherence_matrix(cross_spectral_matrix(X))
    assert np.allclose(np.diag(C), 1.0)


def test_mp_edges_and_eigendecomp():
    lo, hi = mp_edges(4, 16)
    assert lo >= 0.0
    assert hi > lo
    vals, vecs = eigen_decomp(np.eye(3))
    assert vals.shape == (3,)
    assert vecs.shape == (3, 3)


def test_generate_and_snapshot_shapes():
    Y, phi, metadata = generate_mirnov(
        d=6,
        T=64,
        modes=[{"A": 1.0, "omega": 2 * np.pi * 4 / 32, "n": 1, "phase": 0.0}],
        snr_db=0.0,
        seed=2,
    )
    X = fourier_snapshots(Y, L=32, H=32, r=4)
    assert Y.shape == (6, 64)
    assert phi.shape == (6,)
    assert X.shape == (6, 2)
    assert metadata.noise_sigma > 0.0


def test_estimate_n_exact_template():
    phi = toroidal_angles(12)
    v = np.exp(1j * 3 * phi) / np.sqrt(phi.size)
    n_hat, scores = estimate_n(v, phi, np.arange(-5, 6))
    assert n_hat == 3
    assert scores[3] > 0.999
