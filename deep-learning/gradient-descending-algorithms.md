# Gradient Descent Algorithms - Complete Guide

Comprehensive overview of gradient-based optimization algorithms used in training neural networks and deep learning models.

## 📚 Table of Contents

- [Overview](#overview)
- [What is Gradient Descent?](#what-is-gradient-descent)
- [Types of Gradient Descent](#types-of-gradient-descent)
- [Variants & Improvements](#variants--improvements)
- [Mathematical Formulation](#mathematical-formulation)
- [Practical Tips](#practical-tips)
- [Implementation Examples](#implementation-examples)
- [Best Practices](#best-practices)
- [Resources](#resources)

## 🎯 Overview

Gradient descent is the backbone of optimization in deep learning. It is an iterative algorithm used to minimize a loss function by updating model parameters in the direction of the steepest descent (negative gradient). Understanding gradient descent and its variants is crucial for training effective neural networks.

## 🧮 What is Gradient Descent?

Gradient descent is an optimization algorithm that adjusts parameters (weights and biases) to minimize the loss function of a model. At each step, it computes the gradient (partial derivatives) of the loss with respect to each parameter and updates the parameters accordingly.

**Update Rule:**
```
θ_{t+1} = θ_t - α∇J(θ_t)
```

Where:
- `θ_t` are the parameters at iteration t
- `α` is the learning rate
- `∇J(θ_t)` is the gradient of the loss function

### **Intuition**
Think of gradient descent as walking down a hill:
- **Gradient** tells you the steepest direction downhill
- **Learning rate** determines your step size
- **Goal** is to reach the bottom (minimum) of the hill

## 🔄 Types of Gradient Descent

### **1. Batch Gradient Descent (BGD)**
Uses the entire dataset to compute the gradient at each step.

```python
import numpy as np

def batch_gradient_descent(X, y, theta, learning_rate=0.01, max_iterations=1000):
    """
    Batch Gradient Descent implementation
    
    Parameters:
    X: input features (m x n)
    y: target values (m x 1)
    theta: initial parameters (n x 1)
    learning_rate: step size
    max_iterations: maximum number of iterations
    
    Returns:
    theta: optimized parameters
    cost_history: list of cost values
    """
    m = len(y)
    cost_history = []
    
    for iteration in range(max_iterations):
        # Forward pass
        h = np.dot(X, theta)
        
        # Compute cost (Mean Squared Error)
        cost = (1/(2*m)) * np.sum((h - y)**2)
        cost_history.append(cost)
        
        # Compute gradient
        gradient = (1/m) * np.dot(X.T, (h - y))
        
        # Update parameters
        theta = theta - learning_rate * gradient
        
        # Print progress
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Cost: {cost:.6f}")
    
    return theta, cost_history

# Example usage
np.random.seed(42)
X = np.random.randn(100, 3)
y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, 100)
y = y.reshape(-1, 1)

# Add bias term
X_b = np.c_[np.ones((100, 1)), X]

# Initialize parameters
theta = np.random.randn(4, 1)

# Run batch gradient descent
theta_opt, costs = batch_gradient_descent(X_b, y, theta, learning_rate=0.01)

print(f"\nOptimal parameters: {theta_opt.flatten()}")
print(f"Final cost: {costs[-1]:.6f}")
```

**Advantages:**
- Guaranteed convergence to global minimum (for convex functions)
- Stable gradient estimates

**Disadvantages:**
- Computationally expensive for large datasets
- Memory intensive
- Can get stuck in local minima for non-convex functions

### **2. Stochastic Gradient Descent (SGD)**
Uses a single training example to compute the gradient at each step.

```python
def stochastic_gradient_descent(X, y, theta, learning_rate=0.01, max_epochs=100):
    """
    Stochastic Gradient Descent implementation
    
    Parameters:
    X: input features (m x n)
    y: target values (m x 1)
    theta: initial parameters (n x 1)
    learning_rate: step size
    max_epochs: maximum number of epochs
    
    Returns:
    theta: optimized parameters
    cost_history: list of cost values
    """
    m = len(y)
    cost_history = []
    
    for epoch in range(max_epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_cost = 0
        
        for i in range(m):
            # Single example
            x_i = X_shuffled[i:i+1]
            y_i = y_shuffled[i:i+1]
            
            # Forward pass
            h_i = np.dot(x_i, theta)
            
            # Compute cost for this example
            cost_i = (1/2) * (h_i - y_i)**2
            epoch_cost += cost_i[0, 0]
            
            # Compute gradient for this example
            gradient_i = np.dot(x_i.T, (h_i - y_i))
            
            # Update parameters
            theta = theta - learning_rate * gradient_i
        
        # Average cost for this epoch
        avg_cost = epoch_cost / m
        cost_history.append(avg_cost)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Cost: {avg_cost:.6f}")
    
    return theta, cost_history

# Run stochastic gradient descent
theta_opt_sgd, costs_sgd = stochastic_gradient_descent(X_b, y, theta.copy(), learning_rate=0.01)

print(f"\nSGD Optimal parameters: {theta_opt_sgd.flatten()}")
print(f"SGD Final cost: {costs_sgd[-1]:.6f}")
```

**Advantages:**
- Fast updates
- Memory efficient
- Can escape local minima due to noise
- Good for large datasets

**Disadvantages:**
- Noisy gradient estimates
- May not converge to exact minimum
- Requires careful learning rate tuning

### **3. Mini-Batch Gradient Descent**
Uses a small batch of training examples to compute the gradient.

```python
def mini_batch_gradient_descent(X, y, theta, learning_rate=0.01, batch_size=32, max_epochs=100):
    """
    Mini-batch Gradient Descent implementation
    
    Parameters:
    X: input features (m x n)
    y: target values (m x 1)
    theta: initial parameters (n x 1)
    learning_rate: step size
    batch_size: size of mini-batches
    max_epochs: maximum number of epochs
    
    Returns:
    theta: optimized parameters
    cost_history: list of cost values
    """
    m = len(y)
    cost_history = []
    
    for epoch in range(max_epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_cost = 0
        
        # Process mini-batches
        for i in range(0, m, batch_size):
            # Extract mini-batch
            end_idx = min(i + batch_size, m)
            X_batch = X_shuffled[i:end_idx]
            y_batch = y_shuffled[i:end_idx]
            
            # Forward pass
            h_batch = np.dot(X_batch, theta)
            
            # Compute cost for this batch
            batch_cost = (1/(2*len(X_batch))) * np.sum((h_batch - y_batch)**2)
            epoch_cost += batch_cost
            
            # Compute gradient for this batch
            gradient_batch = (1/len(X_batch)) * np.dot(X_batch.T, (h_batch - y_batch))
            
            # Update parameters
            theta = theta - learning_rate * gradient_batch
        
        # Average cost for this epoch
        avg_cost = epoch_cost / (m // batch_size + 1)
        cost_history.append(avg_cost)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Cost: {avg_cost:.6f}")
    
    return theta, cost_history

# Run mini-batch gradient descent
theta_opt_mb, costs_mb = mini_batch_gradient_descent(X_b, y, theta.copy(), learning_rate=0.01, batch_size=16)

print(f"\nMini-batch Optimal parameters: {theta_opt_mb.flatten()}")
print(f"Mini-batch Final cost: {costs_mb[-1]:.6f}")
```

**Advantages:**
- Balance between BGD and SGD
- Better gradient estimates than SGD
- More stable than SGD
- Good for parallelization

**Disadvantages:**
- Requires tuning batch size
- May still get stuck in local minima

## 🚀 Variants & Improvements

### **1. Momentum**
Adds momentum to help accelerate convergence and escape local minima.

```python
def momentum_gradient_descent(X, y, theta, learning_rate=0.01, momentum=0.9, max_iterations=1000):
    """
    Gradient Descent with Momentum
    
    Parameters:
    X: input features (m x n)
    y: target values (m x 1)
    theta: initial parameters (n x 1)
    learning_rate: step size
    momentum: momentum coefficient
    max_iterations: maximum number of iterations
    
    Returns:
    theta: optimized parameters
    cost_history: list of cost values
    """
    m = len(y)
    cost_history = []
    velocity = np.zeros_like(theta)
    
    for iteration in range(max_iterations):
        # Forward pass
        h = np.dot(X, theta)
        
        # Compute cost
        cost = (1/(2*m)) * np.sum((h - y)**2)
        cost_history.append(cost)
        
        # Compute gradient
        gradient = (1/m) * np.dot(X.T, (h - y))
        
        # Update velocity
        velocity = momentum * velocity + learning_rate * gradient
        
        # Update parameters
        theta = theta - velocity
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Cost: {cost:.6f}")
    
    return theta, cost_history

# Run momentum gradient descent
theta_opt_momentum, costs_momentum = momentum_gradient_descent(X_b, y, theta.copy(), learning_rate=0.01)

print(f"\nMomentum Optimal parameters: {theta_opt_momentum.flatten()}")
print(f"Momentum Final cost: {costs_momentum[-1]:.6f}")
```

### **2. Nesterov Accelerated Gradient (NAG)**
Improves momentum by looking ahead to where the momentum will take us.

```python
def nesterov_gradient_descent(X, y, theta, learning_rate=0.01, momentum=0.9, max_iterations=1000):
    """
    Nesterov Accelerated Gradient Descent
    
    Parameters:
    X: input features (m x n)
    y: target values (m x 1)
    theta: initial parameters (n x 1)
    learning_rate: step size
    momentum: momentum coefficient
    max_iterations: maximum number of iterations
    
    Returns:
    theta: optimized parameters
    cost_history: list of cost values
    """
    m = len(y)
    cost_history = []
    velocity = np.zeros_like(theta)
    
    for iteration in range(max_iterations):
        # Look ahead position
        theta_lookahead = theta - momentum * velocity
        
        # Forward pass at lookahead position
        h = np.dot(X, theta_lookahead)
        
        # Compute cost
        cost = (1/(2*m)) * np.sum((h - y)**2)
        cost_history.append(cost)
        
        # Compute gradient at lookahead position
        gradient = (1/m) * np.dot(X.T, (h - y))
        
        # Update velocity
        velocity = momentum * velocity + learning_rate * gradient
        
        # Update parameters
        theta = theta - velocity
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Cost: {cost:.6f}")
    
    return theta, cost_history

# Run Nesterov gradient descent
theta_opt_nag, costs_nag = nesterov_gradient_descent(X_b, y, theta.copy(), learning_rate=0.01)

print(f"\nNAG Optimal parameters: {theta_opt_nag.flatten()}")
print(f"NAG Final cost: {costs_nag[-1]:.6f}")
```

### **3. AdaGrad**
Adapts learning rates for each parameter based on historical gradients.

```python
def adagrad_gradient_descent(X, y, theta, learning_rate=0.01, epsilon=1e-8, max_iterations=1000):
    """
    AdaGrad Gradient Descent
    
    Parameters:
    X: input features (m x n)
    y: target values (m x 1)
    theta: initial parameters (n x 1)
    learning_rate: initial learning rate
    epsilon: small constant to prevent division by zero
    max_iterations: maximum number of iterations
    
    Returns:
    theta: optimized parameters
    cost_history: list of cost values
    """
    m = len(y)
    cost_history = []
    G = np.zeros_like(theta)  # Sum of squared gradients
    
    for iteration in range(max_iterations):
        # Forward pass
        h = np.dot(X, theta)
        
        # Compute cost
        cost = (1/(2*m)) * np.sum((h - y)**2)
        cost_history.append(cost)
        
        # Compute gradient
        gradient = (1/m) * np.dot(X.T, (h - y))
        
        # Accumulate squared gradients
        G += gradient**2
        
        # Update parameters with adaptive learning rate
        theta = theta - (learning_rate / np.sqrt(G + epsilon)) * gradient
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Cost: {cost:.6f}")
    
    return theta, cost_history

# Run AdaGrad gradient descent
theta_opt_adagrad, costs_adagrad = adagrad_gradient_descent(X_b, y, theta.copy(), learning_rate=0.1)

print(f"\nAdaGrad Optimal parameters: {theta_opt_adagrad.flatten()}")
print(f"AdaGrad Final cost: {costs_adagrad[-1]:.6f}")
```

### **4. RMSprop**
Improves AdaGrad by using exponential moving average of squared gradients.

```python
def rmsprop_gradient_descent(X, y, theta, learning_rate=0.01, beta=0.9, epsilon=1e-8, max_iterations=1000):
    """
    RMSprop Gradient Descent
    
    Parameters:
    X: input features (m x n)
    y: target values (m x 1)
    theta: initial parameters (n x 1)
    learning_rate: learning rate
    beta: exponential decay rate
    epsilon: small constant to prevent division by zero
    max_iterations: maximum number of iterations
    
    Returns:
    theta: optimized parameters
    cost_history: list of cost values
    """
    m = len(y)
    cost_history = []
    v = np.zeros_like(theta)  # Moving average of squared gradients
    
    for iteration in range(max_iterations):
        # Forward pass
        h = np.dot(X, theta)
        
        # Compute cost
        cost = (1/(2*m)) * np.sum((h - y)**2)
        cost_history.append(cost)
        
        # Compute gradient
        gradient = (1/m) * np.dot(X.T, (h - y))
        
        # Update moving average of squared gradients
        v = beta * v + (1 - beta) * gradient**2
        
        # Update parameters
        theta = theta - (learning_rate / np.sqrt(v + epsilon)) * gradient
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Cost: {cost:.6f}")
    
    return theta, cost_history

# Run RMSprop gradient descent
theta_opt_rmsprop, costs_rmsprop = rmsprop_gradient_descent(X_b, y, theta.copy(), learning_rate=0.01)

print(f"\nRMSprop Optimal parameters: {theta_opt_rmsprop.flatten()}")
print(f"RMSprop Final cost: {costs_rmsprop[-1]:.6f}")
```

### **5. Adam (Adaptive Moment Estimation)**
Combines the benefits of momentum and RMSprop.

```python
def adam_gradient_descent(X, y, theta, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iterations=1000):
    """
    Adam Gradient Descent
    
    Parameters:
    X: input features (m x n)
    y: target values (m x 1)
    theta: initial parameters (n x 1)
    learning_rate: learning rate
    beta1: exponential decay rate for first moment
    beta2: exponential decay rate for second moment
    epsilon: small constant to prevent division by zero
    max_iterations: maximum number of iterations
    
    Returns:
    theta: optimized parameters
    cost_history: list of cost values
    """
    m = len(y)
    cost_history = []
    m_t = np.zeros_like(theta)  # First moment (mean)
    v_t = np.zeros_like(theta)  # Second moment (variance)
    
    for iteration in range(max_iterations):
        # Forward pass
        h = np.dot(X, theta)
        
        # Compute cost
        cost = (1/(2*m)) * np.sum((h - y)**2)
        cost_history.append(cost)
        
        # Compute gradient
        gradient = (1/m) * np.dot(X.T, (h - y))
        
        # Update biased first moment estimate
        m_t = beta1 * m_t + (1 - beta1) * gradient
        
        # Update biased second moment estimate
        v_t = beta2 * v_t + (1 - beta2) * gradient**2
        
        # Bias correction
        m_t_corrected = m_t / (1 - beta1**(iteration + 1))
        v_t_corrected = v_t / (1 - beta2**(iteration + 1))
        
        # Update parameters
        theta = theta - (learning_rate / np.sqrt(v_t_corrected + epsilon)) * m_t_corrected
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Cost: {cost:.6f}")
    
    return theta, cost_history

# Run Adam gradient descent
theta_opt_adam, costs_adam = adam_gradient_descent(X_b, y, theta.copy(), learning_rate=0.01)

print(f"\nAdam Optimal parameters: {theta_opt_adam.flatten()}")
print(f"Adam Final cost: {costs_adam[-1]:.6f}")
```

## 📐 Mathematical Formulation

### **Gradient Computation**
For a loss function J(θ), the gradient is:
```
∇J(θ) = [∂J/∂θ₁, ∂J/∂θ₂, ..., ∂J/∂θₙ]ᵀ
```

### **Update Rules**

#### **Standard Gradient Descent:**
```
θ_{t+1} = θ_t - α∇J(θ_t)
```

#### **Momentum:**
```
v_{t+1} = βv_t + α∇J(θ_t)
θ_{t+1} = θ_t - v_{t+1}
```

#### **AdaGrad:**
```
G_t = G_{t-1} + ∇J(θ_t)²
θ_{t+1} = θ_t - (α/√(G_t + ε))∇J(θ_t)
```

#### **RMSprop:**
```
v_t = βv_{t-1} + (1-β)∇J(θ_t)²
θ_{t+1} = θ_t - (α/√(v_t + ε))∇J(θ_t)
```

#### **Adam:**
```
m_t = β₁m_{t-1} + (1-β₁)∇J(θ_t)
v_t = β₂v_{t-1} + (1-β₂)∇J(θ_t)²
m̂_t = m_t/(1-β₁ᵗ)
v̂_t = v_t/(1-β₂ᵗ)
θ_{t+1} = θ_t - (α/√(v̂_t + ε))m̂_t
```

## 💡 Practical Tips

### **1. Learning Rate Selection**
- **Too high**: May cause divergence
- **Too low**: Slow convergence
- **Rule of thumb**: Start with 0.01 and adjust

### **2. Learning Rate Scheduling**
```python
def learning_rate_scheduler(initial_lr, decay_rate, epoch):
    """Exponential learning rate decay"""
    return initial_lr * (1 / (1 + decay_rate * epoch))

def step_lr_scheduler(initial_lr, decay_factor, decay_epochs):
    """Step learning rate decay"""
    def get_lr(epoch):
        for decay_epoch in decay_epochs:
            if epoch >= decay_epoch:
                initial_lr *= decay_factor
        return initial_lr
    return get_lr
```

### **3. Gradient Clipping**
```python
def clip_gradients(gradients, max_norm):
    """Clip gradients to prevent exploding gradients"""
    norm = np.linalg.norm(gradients)
    if norm > max_norm:
        gradients = gradients * max_norm / norm
    return gradients
```

## 💻 Implementation Examples

### **Complete Training Loop with Multiple Optimizers**

```python
import matplotlib.pyplot as plt

def compare_optimizers(X, y, initial_theta, max_iterations=1000):
    """Compare different optimization algorithms"""
    
    # Run different optimizers
    theta_bgd, costs_bgd = batch_gradient_descent(X, y, initial_theta.copy(), max_iterations=max_iterations)
    theta_sgd, costs_sgd = stochastic_gradient_descent(X, y, initial_theta.copy(), max_epochs=max_iterations//10)
    theta_momentum, costs_momentum = momentum_gradient_descent(X, y, initial_theta.copy(), max_iterations=max_iterations)
    theta_adam, costs_adam = adam_gradient_descent(X, y, initial_theta.copy(), max_iterations=max_iterations)
    
    # Plot convergence
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(costs_bgd, label='Batch GD')
    plt.title('Batch Gradient Descent')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(costs_sgd, label='Stochastic GD')
    plt.title('Stochastic Gradient Descent')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(costs_momentum, label='Momentum GD')
    plt.title('Gradient Descent with Momentum')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(costs_adam, label='Adam')
    plt.title('Adam Optimizer')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print final results
    print("Final Cost Comparison:")
    print(f"Batch GD: {costs_bgd[-1]:.6f}")
    print(f"Stochastic GD: {costs_sgd[-1]:.6f}")
    print(f"Momentum GD: {costs_momentum[-1]:.6f}")
    print(f"Adam: {costs_adam[-1]:.6f}")
    
    return {
        'batch_gd': (theta_bgd, costs_bgd),
        'stochastic_gd': (theta_sgd, costs_sgd),
        'momentum_gd': (theta_momentum, costs_momentum),
        'adam': (theta_adam, costs_adam)
    }

# Compare all optimizers
results = compare_optimizers(X_b, y, theta.copy(), max_iterations=500)
```

### **Neural Network Training with Adam**

```python
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))
        
        # Adam parameters
        self.m_W1 = np.zeros_like(self.W1)
        self.v_W1 = np.zeros_like(self.W1)
        self.m_b1 = np.zeros_like(self.b1)
        self.v_b1 = np.zeros_like(self.b1)
        self.m_W2 = np.zeros_like(self.W2)
        self.v_W2 = np.zeros_like(self.W2)
        self.m_b2 = np.zeros_like(self.b2)
        self.v_b2 = np.zeros_like(self.b2)
        
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.z1 = np.dot(self.W1, X.T) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2.T
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        
        # Compute gradients
        dz2 = self.a2.T - y.reshape(-1, 1)
        dW2 = np.dot(dz2, self.a1.T) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        dz1 = np.dot(self.W2.T, dz2) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(dz1, X) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        
        # Adam updates
        self.t += 1
        
        # Update W1
        self.m_W1 = self.beta1 * self.m_W1 + (1 - self.beta1) * dW1
        self.v_W1 = self.beta2 * self.v_W1 + (1 - self.beta2) * dW1**2
        m_W1_corrected = self.m_W1 / (1 - self.beta1**self.t)
        v_W1_corrected = self.v_W1 / (1 - self.beta2**self.t)
        self.W1 -= learning_rate * m_W1_corrected / (np.sqrt(v_W1_corrected) + self.epsilon)
        
        # Update b1
        self.m_b1 = self.beta1 * self.m_b1 + (1 - self.beta1) * db1
        self.v_b1 = self.beta2 * self.v_b1 + (1 - self.beta2) * db1**2
        m_b1_corrected = self.m_b1 / (1 - self.beta1**self.t)
        v_b1_corrected = self.v_b1 / (1 - self.beta2**self.t)
        self.b1 -= learning_rate * m_b1_corrected / (np.sqrt(v_b1_corrected) + self.epsilon)
        
        # Update W2
        self.m_W2 = self.beta1 * self.m_W2 + (1 - self.beta1) * dW2
        self.v_W2 = self.beta2 * self.v_W2 + (1 - self.beta2) * dW2**2
        m_W2_corrected = self.m_W2 / (1 - self.beta1**self.t)
        v_W2_corrected = self.v_W2 / (1 - self.beta2**self.t)
        self.W2 -= learning_rate * m_W2_corrected / (np.sqrt(v_W2_corrected) + self.epsilon)
        
        # Update b2
        self.m_b2 = self.beta1 * self.m_b2 + (1 - self.beta1) * db2
        self.v_b2 = self.beta2 * self.v_b2 + (1 - self.beta2) * db2**2
        m_b2_corrected = self.m_b2 / (1 - self.beta1**self.t)
        v_b2_corrected = self.v_b2 / (1 - self.beta2**self.t)
        self.b2 -= learning_rate * m_b2_corrected / (np.sqrt(v_b2_corrected) + self.epsilon)
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        costs = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute cost
            cost = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
            costs.append(cost)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.6f}")
        
        return costs

# Example usage for binary classification
X_class = np.random.randn(100, 2)
y_class = (X_class[:, 0] + X_class[:, 1] > 0).astype(int).reshape(-1, 1)

# Create and train neural network
nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
costs = nn.train(X_class, y_class, epochs=500, learning_rate=0.01)

# Plot training progress
plt.figure(figsize=(10, 6))
plt.plot(costs)
plt.title('Neural Network Training with Adam')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.grid(True)
plt.show()
```

## 💡 Best Practices

### **1. Algorithm Selection**
- **SGD**: Good starting point, simple to implement
- **Momentum**: Better convergence, good for most cases
- **Adam**: Excellent default choice, adaptive learning rates
- **AdaGrad/RMSprop**: Good for sparse data

### **2. Hyperparameter Tuning**
- **Learning rate**: Most important parameter
- **Batch size**: Balance between speed and stability
- **Momentum**: Usually 0.9 works well
- **Beta values**: Adam defaults (0.9, 0.999) are good

### **3. Monitoring Training**
- Watch for convergence
- Monitor gradient norms
- Use validation set for early stopping
- Implement learning rate scheduling

### **4. Common Issues**
- **Vanishing gradients**: Use ReLU, proper initialization
- **Exploding gradients**: Use gradient clipping
- **Local minima**: Use momentum, multiple restarts
- **Overfitting**: Use regularization, early stopping

## 📚 Resources

### **Books**
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- "Neural Networks and Deep Learning" by Michael Nielsen
- "Optimization for Machine Learning" by Suvrit Sra

### **Online Courses**
- [Stanford CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [MIT 6.S191: Introduction to Deep Learning](http://introtodeeplearning.com/)
- [Coursera Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)

### **Research Papers**
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/)
- [On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237)

### **Python Libraries**
- [TensorFlow](https://www.tensorflow.org/) - Built-in optimizers
- [PyTorch](https://pytorch.org/) - Comprehensive optimizer implementations
- [Keras](https://keras.io/) - High-level optimizer API
- [NumPy](https://numpy.org/) - For custom implementations

### **Blogs**
- [An overview of gradient descent optimization algorithms (Sebastian Ruder's blog)](https://www.ruder.io/optimizing-gradient-descent/)

---

**Happy Optimization Learning! 🚀✨**

*Gradient descent algorithms are the engines that power deep learning. Understanding their strengths, weaknesses, and proper usage is essential for training effective neural networks.*