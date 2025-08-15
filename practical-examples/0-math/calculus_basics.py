
"""
Calculus Essentials for Deep Learning
------------------------------------
Numeric intuition only (no sympy) to keep dependencies light.
Topics:
  - Derivatives & the chain rule via finite differences
  - Gradients of multivariate functions
  - Gradient descent on a simple function
  - Automatic differentiation intuition (manual backprop for a tiny net)
"""

import numpy as np

def finite_difference_derivative(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2*h)

def gradient(f, x, h=1e-6):
    """Approximate gradient of f: R^n -> R at point x."""
    x = np.array(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        ei = np.zeros_like(x); ei[i] = 1.0
        g[i] = (f(x + h*ei) - f(x - h*ei)) / (2*h)
    return g

def demo_derivatives():
    f = lambda x: x**3 + 2*x**2 - 5*x
    x0 = 1.5
    approx = finite_difference_derivative(f, x0)
    true = 3*x0**2 + 4*x0 - 5
    print("f'(x0) approx:", approx, " true:", true)
    return approx, true

def demo_gradient_descent():
    # Quadratic bowl: f(x,y) = (x-3)^2 + (y+2)^2
    f = lambda z: (z[0]-3)**2 + (z[1]+2)**2
    z = np.array([10.0, -10.0])
    lr = 0.1
    for t in range(100):
        g = gradient(f, z)
        z -= lr * g
    print("Min approx at:", z, " f(z)=", f(z))
    return z

def tiny_net_forward(x, params):
    """One-hidden-layer net: y = ReLU(xW1+b1) W2 + b2"""
    W1, b1, W2, b2 = params
    h = np.maximum(0.0, x @ W1 + b1)   # ReLU
    y = h @ W2 + b2
    cache = (x, h, params)
    return y, cache

def tiny_net_backward(dy, cache):
    """Manual backprop for the tiny net above (single sample)."""
    x, h, (W1, b1, W2, b2) = cache
    dh = dy * W2
    drelu = dh * (h > 0).astype(float)
    dW2 = np.outer(h, dy)
    db2 = dy
    dW1 = np.outer(x, drelu)
    db1 = drelu
    dx = drelu @ W1.T
    return (dW1, db1, dW2, db2), dx

def demo_manual_backprop(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(3,))             # input dim=3
    W1 = rng.normal(size=(3, 4))          # hidden=4
    b1 = rng.normal(size=(4,))
    W2 = rng.normal(size=(4,))            # output scalar
    b2 = rng.normal()
    params = (W1, b1, W2, b2)

    y, cache = tiny_net_forward(x, params)
    # loss: L = 0.5*(y - y_target)^2
    y_target = 1.0
    dy = (y - y_target)  # dL/dy
    grads, dx = tiny_net_backward(dy, cache)

    print("y:", float(y), " dL/dy:", float(dy))
    for name, g in zip(["dW1","db1","dW2","db2"], grads):
        print(name, "shape", np.shape(g))
    return grads, dx

def main():
    print("== Calculus Essentials ==")
    demo_derivatives()
    demo_gradient_descent()
    demo_manual_backprop()

if __name__ == "__main__":
    main()
