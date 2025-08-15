# Linear Algebra Essentials for Deep Learning - Code Explanation

This document explains the `linear_algebra_basics.py` file, which demonstrates fundamental linear algebra concepts essential for understanding deep learning algorithms.

## Overview

The code covers core linear algebra operations using only NumPy, making it lightweight and educational. It demonstrates:

- **Vectors & matrices** - Basic data structures
- **Dot products & matrix multiplication** - Core operations
- **Norms & distances** - Measuring vector magnitudes and separations
- **Projections** - Vector decomposition
- **Orthogonality** - Perpendicular relationships
- **Linear systems & least squares** - Solving equations
- **Eigenvalues/eigenvectors & PCA intuition** - Dimensionality reduction

## Core Functions Explained

### 1. Vectors and Matrices

```python
def vectors_and_matrices():
    v = np.array([1, 2, 3])             # vector (3,)
    w = np.array([4, 5, 6])
    A = np.array([[1, 2], [3, 4], [5, 6]])  # matrix (3x2)
    print("v:", v)
    print("w:", w)
    print("A:\n", A)
    return v, w, A
```

**What it does:** Creates and displays basic vector and matrix objects.

**Key concepts:**
- **Vector `v`**: 1D array with shape `(3,)` representing a point in 3D space
- **Vector `w`**: Another 3D vector
- **Matrix `A`**: 2D array with shape `(3,2)` representing a linear transformation

**Shape notation:**
- `(3,)` means 3 elements in 1 dimension
- `(3,2)` means 3 rows and 2 columns

### 2. Dot Products and Matrix Multiplication

```python
def dot_and_matmul(v, w, A):
    dot = v @ w              # dot product
    B = A.T @ A              # matrix multiplication
    print("v·w =", dot)
    print("A^T A =\n", B)
    # quick checks
    assert np.isclose(dot, np.sum(v*w))
    assert B.shape == (2,2)
    return dot, B
```

**What it does:** Demonstrates vector dot products and matrix multiplication.

**Dot product `v @ w`:**
- **Mathematical formula:** `v·w = v₁w₁ + v₂w₂ + v₃w₃`
- **Geometric meaning:** `v·w = ||v|| × ||w|| × cos(θ)` where θ is the angle between vectors
- **Alternative computation:** `np.sum(v*w)` (element-wise multiplication then sum)

**Matrix multiplication `A.T @ A`:**
- **A.T**: Transpose of A (flips rows and columns)
- **Result shape:** `(2,3) @ (3,2) = (2,2)` (inner dimensions must match)
- **What it computes:** `A^T A` is a symmetric matrix that appears in least squares problems

### 3. Norms and Distances

```python
def norms_and_distances(v, w):
    l2_v = np.linalg.norm(v, ord=2)
    l1_v = np.linalg.norm(v, ord=1)
    dist_vw = np.linalg.norm(v - w)
    print("||v||2 =", l2_v, " ||v||1 =", l1_v, " dist(v,w) =", dist_vw)
    return l2_v, l1_v, dist_vw
```

**What it does:** Computes different types of vector norms and distances.

**L2 norm (Euclidean norm):**
- **Formula:** `||v||₂ = √(v₁² + v₂² + v₃²)`
- **Geometric meaning:** Length of the vector from origin to point v
- **Most common:** Used in most machine learning applications

**L1 norm (Manhattan norm):**
- **Formula:** `||v||₁ = |v₁| + |v₂| + |v₃|`
- **Geometric meaning:** Sum of absolute values (like walking in a grid city)
- **Use cases:** L1 regularization (Lasso), sparse solutions

**Distance between vectors:**
- **Formula:** `dist(v,w) = ||v - w||₂`
- **Geometric meaning:** Euclidean distance between two points
- **Computation:** `np.linalg.norm(v - w)` where `v - w` is the difference vector

### 4. Vector Projection

```python
def projection(u, v):
    """Project u onto v (both 1-D arrays)."""
    denom = v @ v
    if np.isclose(denom, 0):
        raise ValueError("Cannot project onto zero vector")
    coef = (u @ v) / denom
    proj = coef * v
    print("proj_v(u) =", proj, " coefficient =", coef)
    return proj
```

**What it does:** Projects vector `u` onto vector `v`.

**Mathematical formula:** `proj_v(u) = ((u·v)/(v·v)) × v`

**How it works:**
1. **Coefficient calculation:** `coef = (u·v)/(v·v)` measures how much of u lies in v's direction
2. **Projection:** `proj = coef × v` gives the component of u that's parallel to v
3. **Geometric meaning:** The "shadow" of u cast onto v

**Why this matters:**
- **Dimensionality reduction:** Project high-dimensional data onto principal directions
- **Signal processing:** Extract components in specific directions
- **Machine learning:** Feature extraction and compression

### 5. Orthogonality

```python
def orthogonality(u, v):
    dot = u @ v
    print("u·v =", dot, "(≈0 means orthogonal)")
    return dot
```

**What it does:** Checks if two vectors are orthogonal (perpendicular).

**Mathematical condition:** Vectors u and v are orthogonal if `u·v = 0`

**Why orthogonality matters:**
- **Basis vectors:** Orthogonal vectors can form coordinate systems
- **Principal components:** PCA finds orthogonal directions of maximum variance
- **Signal separation:** Orthogonal signals don't interfere with each other

### 6. Least Squares Demo

```python
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
```

**What it does:** Solves the overdetermined system `Ax ≈ b` using least squares.

**The problem:**
- **A**: 100×3 matrix (more equations than unknowns)
- **b**: 100-dimensional vector
- **Goal**: Find x that minimizes `||Ax - b||²`

**Two solution methods:**

1. **Normal equations:** `(A^T A) x = A^T b`
   - **Pros:** Simple to understand
   - **Cons:** Can be numerically unstable if A^T A is ill-conditioned

2. **NumPy lstsq:** Uses more sophisticated algorithms (SVD, QR decomposition)
   - **Pros:** More numerically stable
   - **Cons:** Black box implementation

**Why least squares matters:**
- **Linear regression:** Finding best-fit line through data points
- **Neural networks:** Training involves solving many least squares problems
- **Data fitting:** Approximating complex relationships with simple models

### 7. Eigenvalues, Eigenvectors, and PCA Intuition

```python
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
```

**What it does:** Demonstrates eigenvalue decomposition and PCA concepts.

**Data generation:**
- **X**: 500 points from anisotropic 2D Gaussian
- **Transformation matrix**: `[[3.0, 0.0], [0.0, 0.5]]` stretches data 3x in x-direction, 0.5x in y-direction

**Covariance matrix C:**
- **Formula:** `C_ij = E[(X_i - μ_i)(X_j - μ_j)]`
- **Geometric meaning:** Describes how data varies in different directions
- **Properties:** Symmetric, positive semi-definite

**Eigenvalue decomposition:**
- **Eigenvalues:** Measure variance in principal directions
- **Eigenvectors:** Point in directions of maximum variance
- **Sorting:** Largest eigenvalue corresponds to direction of most variation

**PCA intuition:**
- **First principal component:** Direction of maximum variance (along the "stretch")
- **Second principal component:** Direction of remaining variance (perpendicular to first)
- **Dimensionality reduction:** Project data onto top k principal components

## Key Concepts for Deep Learning

### Why Linear Algebra Matters

1. **Data representation:** Vectors and matrices represent features and samples
2. **Transformations:** Linear layers in neural networks are matrix multiplications
3. **Optimization:** Gradients are vectors, Hessians are matrices
4. **Dimensionality reduction:** PCA, t-SNE, and autoencoders rely on linear algebra

### Common Operations in Deep Learning

- **Matrix multiplication:** Forward pass through linear layers
- **Transpose operations:** Backpropagation gradients
- **Eigenvalue decomposition:** Principal component analysis
- **Least squares:** Linear regression and optimization
- **Norms:** Loss functions and regularization

### Numerical Considerations

- **Conditioning:** Some matrices are numerically unstable
- **Precision:** Float32 vs float64 trade-offs
- **Memory:** Matrix operations can be memory-intensive
- **Efficiency:** BLAS/LAPACK libraries optimize common operations

## Running the Code

The main function demonstrates all concepts:

```python
def main():
    print("== Linear Algebra Essentials ==")
    v, w, A = vectors_and_matrices()     # Basic objects
    dot_and_matmul(v, w, A)              # Core operations
    norms_and_distances(v, w)             # Vector measurements
    projection(u=np.array([3.0, 1.0]), v=np.array([2.0, 0.0]))  # Projections
    orthogonality(np.array([1,0,0]), np.array([0,1,0]))          # Orthogonality
    least_squares_demo()                  # Solving equations
    eigen_and_pca_intuition()            # Dimensionality reduction
```

## Extensions and Learning Path

1. **Try different vector operations** with various shapes and sizes
2. **Experiment with different matrix types** (symmetric, diagonal, orthogonal)
3. **Implement matrix factorization methods** (LU, QR, SVD)
4. **Explore applications in computer vision** (image transformations, filters)
5. **Study advanced topics** like tensor operations and group theory

## Practical Applications

- **Computer Vision:** Image processing, convolutional operations
- **Natural Language Processing:** Word embeddings, attention mechanisms
- **Reinforcement Learning:** Value function approximation, policy gradients
- **Graph Neural Networks:** Adjacency matrices, spectral methods
- **Quantum Machine Learning:** Quantum state representations

This code serves as a foundation for understanding the mathematical operations that power modern deep learning systems, from simple linear transformations to complex dimensionality reduction techniques.
