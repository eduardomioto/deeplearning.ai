
"""
Linear Algebra Essentials for Deep Learning
------------------------------------------
Run this file directly (python linear_algebra_basics.py) to see demos.
Uses only numpy.
Topics:
  - Vectors & matrices
  - Dot product & matrix multiplication
  - Norms & distances
  - Projections
  - Orthogonality
  - Linear systems & least squares
  - Eigenvalues / eigenvectors & PCA intuition
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

def vectors_and_matrices():
    v = np.array([1, 2, 3])             # vector (3,)
    w = np.array([4, 5, 6])
    A = np.array([[1, 2], [3, 4], [5, 6]])  # matrix (3x2)
    print("v:", v)
    print("w:", w)
    print("A:\n", A)
    return v, w, A

def dot_and_matmul(v, w, A):
    dot = v @ w              # dot product
    B = A.T @ A              # matrix multiplication
    print("v·w =", dot)
    print("A^T A =\n", B)
    # quick checks
    assert np.isclose(dot, np.sum(v*w))
    assert B.shape == (2,2)
    return dot, B

def norms_and_distances(v, w):
    l2_v = np.linalg.norm(v, ord=2)
    l1_v = np.linalg.norm(v, ord=1)
    dist_vw = np.linalg.norm(v - w)
    print("||v||2 =", l2_v, " ||v||1 =", l1_v, " dist(v,w) =", dist_vw)
    return l2_v, l1_v, dist_vw

def projection(u, v):
    """Project u onto v (both 1-D arrays)."""
    denom = v @ v
    if np.isclose(denom, 0):
        raise ValueError("Cannot project onto zero vector")
    coef = (u @ v) / denom
    proj = coef * v
    print("proj_v(u) =", proj, " coefficient =", coef)
    return proj

def orthogonality(u, v):
    dot = u @ v
    print("u·v =", dot, "(≈0 means orthogonal)")
    return dot

def least_squares_demo(seed=0):
    """Solve Ax≈b using normal equations and compare with lstsq."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(100, 3))
    true_x = np.array([2.0, -1.0, 0.5])
    b = A @ true_x + rng.normal(scale=0.1, size=100)
    # Normal equation: (A^T A) x = A^T b
    x_ne = np.linalg.solve(A.T @ A, A.T @ b)
    x_lstsq, *_ = np.linalg.lstsq(A, b, rcond=None)
    print("True x:", true_x)
    print("NormalEq x:", x_ne)
    print("lstsq x:", x_lstsq)
    return true_x, x_ne, x_lstsq

def eigen_and_pca_intuition(seed=1):
    """Compute eigenvalues/eigenvectors and show principal direction."""
    rng = np.random.default_rng(seed)
    # Anisotropic 2D Gaussian
    X = rng.normal(size=(500, 2)) @ np.array([[3.0, 0.0],[0.0, 0.5]])
    # Covariance
    C = np.cov(X.T)
    vals, vecs = np.linalg.eig(C)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    print("Covariance:\n", C)
    print("Eigenvalues:", vals)
    print("Principal component (first eigenvector):", vecs[:,0])
    # Verify orthonormal eigenvectors for symmetric C
    I = vecs.T @ vecs
    assert np.allclose(I, np.eye(2), atol=1e-6)
    return C, vals, vecs

def main():
    print("== Linear Algebra Essentials ==")
    v, w, A = vectors_and_matrices()
    dot_and_matmul(v, w, A)
    norms_and_distances(v, w)
    projection(u=np.array([3.0, 1.0]), v=np.array([2.0, 0.0]))
    orthogonality(np.array([1,0,0]), np.array([0,1,0]))
    least_squares_demo()
    eigen_and_pca_intuition()

if __name__ == "__main__":
    main()
