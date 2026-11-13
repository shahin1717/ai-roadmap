# Assignment: How Small Should the Step Be?

A derivative is a limit, so a computer can approximate it with a small step $h$. In this assignment you measure how the error of that approximation behaves as $h$ shrinks, find the best step size, and explain what you see. It takes about 45-60 minutes and needs only NumPy and Matplotlib. Back to the [topic](README.md).

## The setup

You will test three functions whose derivatives you know exactly, so that every error can be measured. The numbers are invented for this exercise.

```python
import numpy as np

sigmoid = lambda z: 1 / (1 + np.exp(-z))

# name: (function, its exact derivative, the point x0)
tests = {
    "exp at 1":     (np.exp, np.exp, 1.0),
    "ln at 2":      (np.log, lambda x: 1 / x, 2.0),
    "sigmoid at 3": (sigmoid, lambda z: sigmoid(z) * (1 - sigmoid(z)), 3.0),
}
hs = np.logspace(-1, -16, 31)         # step sizes from 1e-1 down to 1e-16
```

Two approximations of the slope of $f$ at $x_0$ with a step $h$:

```math
\text{forward: } \frac{f(x_0+h) - f(x_0)}{h}, \qquad \text{centered: } \frac{f(x_0+h) - f(x_0-h)}{2h}.
```

## Tasks

1. **Compute the errors.** For each test function, compute the forward and the centered approximation for every $h$ in `hs` (use NumPy's vectorized arithmetic on the whole array `hs` at once), and the absolute error against the exact derivative.
2. **Plot.** Draw two log-log plots side by side, one for forward errors and one for centered errors, with one line per test function and the step axis reversed so that $h$ shrinks to the right. Label the axes and add a legend.
3. **Find the sweet spot.** For each function and each method, print the step size with the smallest error, and that smallest error. Compare with $\sqrt{\varepsilon}$ and $\varepsilon^{1/3}$, where $\varepsilon$ is `np.finfo(float).eps`. In one sentence each, say which step size is best for the forward method and for the centered method.
4. **Measure the order.** On the part of the plot where the error falls (use $10^{-4} \le h \le 10^{-1}$), fit a straight line to $\log_{10}(\text{error})$ against $\log_{10}(h)$ with `np.polyfit(..., 1)`. The slope tells you how fast the error shrinks: a slope of $1$ means the error is proportional to $h$, a slope of $2$ means proportional to $h^2$. What slopes do you get for the two methods?
5. **Explain the rise.** For $f = \exp$ at $x_0 = 1$, print $f(x_0 + h) - f(x_0)$ for $h = 10^{-4}, 10^{-8}, 10^{-12}, 10^{-15}$. Compare each difference with $\varepsilon \cdot f(x_0)$, the size of the rounding error in a stored value of $f$. In three or four sentences, explain why the total error goes up again for tiny $h$ even though the limit says it should keep improving.
6. **A gradient check.** Take the tiny loss $L(\mathbf{w}) = (\sigma(\mathbf{w}\cdot\mathbf{x}) - y)^2$ with `x = [0.8, -1.5, 0.4]`, `y = 0.3`, and `w0 = [0.2, -0.1, 0.5]`, where $\sigma$ is the sigmoid. Its exact gradient is $2(s - y)\,s(1-s)\,\mathbf{x}$ with $s = \sigma(\mathbf{w}\cdot\mathbf{x})$. Compute the centered finite-difference gradient (one coordinate at a time) for $h = 10^{-2}, 10^{-5}, 10^{-8}, 10^{-12}$ and print the relative error $\lVert \text{numeric} - \text{exact}\rVert / \lVert \text{numeric} + \text{exact}\rVert$ for each. Which $h$ would you use for a gradient check, and why?
7. **Report.** In 4-6 sentences, as if to a teammate who wants to check a hand-written gradient: which method and step size to use, what error to expect, why not to choose the smallest possible $h$, and one thing that surprised you.

## What to hand in

A short notebook or script that runs top to bottom, with the plot, plus your written explanations for tasks 3, 5, and 7.

## Stretch (optional)

Repeat the centered experiment with 32-bit floats (convert $x_0$ and `hs` with `.astype(np.float32)`, using steps from $10^{-1}$ to $10^{-7}$). Where does the sweet spot move, and does the new location fit $\varepsilon^{1/3}$ for 32-bit floats (`np.finfo(np.float32).eps`)? What does this suggest for training in lower precision?
