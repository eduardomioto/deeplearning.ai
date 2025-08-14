

Weights (w) → how important each input is

Bias (w₀) → a baseline adjustment

Activation function (g) → a way to introduce non-linear thinking (so it’s not just a straight line formula)

In plain English:

Each circle takes numbers from the previous layer, multiplies them by some learned importance values, adds a little adjustment, and then decides how “active” it should be.

How it works in practice

You give the network data (blue).

It processes it step-by-step through the hidden layers (red).

You get a prediction or result (purple).

If it’s wrong, it adjusts the “weights” so next time it’s closer to correct.

This is repeated thousands or millions of times until the network is good at its job.