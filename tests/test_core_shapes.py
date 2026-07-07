import numpy as np

from mirnov_rmt.geometry import toroidal_angles
from mirnov_rmt.rmt import eigen_decomp, mp_edges
from mirnov_rmt.spectral import coherence_matrix, cross_spectral_matrix


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
