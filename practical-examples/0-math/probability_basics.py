
"""
Probability Essentials for Deep Learning
---------------------------------------
Topics:
  - Random variables & distributions (sampling)
  - Expectation & variance (Monte Carlo estimation)
  - Law of Large Numbers (LLN) demo
  - Bayes rule with a simple diagnostic test
"""

import numpy as np

def sample_distributions(seed=0, n=10000):
    rng = np.random.default_rng(seed)
    normals = rng.normal(loc=0.0, scale=1.0, size=n)
    uniforms = rng.uniform(low=-1.0, high=1.0, size=n)
    bernoulli = rng.binomial(1, 0.3, size=n)
    print("samples -> normal mean≈", normals.mean(), " uniform mean≈", uniforms.mean(), " bernoulli p̂≈", bernoulli.mean())
    return normals, uniforms, bernoulli

def expectation_variance(samples):
    mean = np.mean(samples)
    var = np.var(samples, ddof=0)
    print("E[X]≈", mean, " Var[X]≈", var)
    return mean, var

def law_of_large_numbers(seed=1, n=100000):
    rng = np.random.default_rng(seed)
    p = 0.37
    xs = rng.binomial(1, p, size=n)
    running_means = np.cumsum(xs) / np.arange(1, n+1)
    print("Final running mean (should approach p=0.37):", running_means[-1])
    return running_means

def bayes_rule_example(prior=0.01, sens=0.95, spec=0.98):
    """Posteriors for a positive test: P(Disease|Positive)."""
    p_pos = prior*sens + (1-prior)*(1-spec)
    posterior = (prior*sens) / p_pos
    print(f"Prior={prior}, Sens={sens}, Spec={spec} -> P(Disease|Positive)≈ {posterior:.4f}")
    return posterior

def main():
    print("== Probability Essentials ==")
    normals, uniforms, bernoulli = sample_distributions()
    expectation_variance(normals)
    law_of_large_numbers()
    bayes_rule_example()

if __name__ == "__main__":
    main()
