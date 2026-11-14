# Assignment: Normalising a Density by Numerical Integration

A model gives you a "waiting-time score" curve that is not yet a probability density: it is positive, but its area is not $1$. In this assignment you build three numerical integration rules from scratch, measure how fast their errors shrink, and then use them to normalise the curve, compute its mean and some probabilities, and check everything against SciPy and an exact answer. It takes about 60 minutes and needs only NumPy, Matplotlib, SciPy, and SymPy. Back to the [topic](README.md).

## The setup

The unnormalised curve is invented for this exercise:

```python
import numpy as np

g = lambda x: x**2 * np.exp(-x)       # defined for x >= 0; it is tiny beyond x = 40
```

A density $f$ is $g$ divided by its total area $Z = \int_0^\infty g(x)\,dx$. You cannot integrate to infinity on a computer with a fixed grid, so you will integrate over $[0, 40]$ (the part beyond $40$ is far below rounding error here) and compare with SciPy's infinite-range `quad`.

## Tasks

1. **Build the rules.** Write `midpoint(f, a, b, n)`, `trapezoid_rule(f, a, b, n)`, and `simpson_rule(f, a, b, n)` using vectorized NumPy (no Python loops over strips). Make `simpson_rule` raise a clear error when $n$ is odd.
2. **Measure the errors.** Use the exact value of $\int_0^3 g(x)\,dx$ from SymPy. For $n = 4, 8, 16, 32, 64, 128$, compute the absolute error of each rule. Draw one log-log plot of error against $n$ (label axes, add a legend), and fit a straight line to $\log(\text{error})$ against $\log(n)$ for each rule with `np.polyfit`. What slopes do you get, and how do they match the theory ($-2$, $-2$, $-4$)?
3. **Normalise.** Compute $Z$ with `simpson_rule` on $[0, 40]$ for $n = 100, 400, 1600$. Does it settle down, and to what? Compare with `scipy.integrate.quad(g, 0, np.inf)` using `np.isclose`. Define the density $f(x) = g(x)/Z$ and confirm that its area is $1$.
4. **Mean and probabilities.** Using your rules, compute $E[X] = \int_0^{40} x\,f(x)\,dx$, $P(X \le 2)$, and $P(1 \le X \le 3)$. Check each against `quad`. Then use SymPy to get exact values (the exact $Z$ is a whole number, and the exact mean is too), and print the absolute differences.
5. **The CDF and the median.** Build the CDF on a grid of $4001$ points over $[0, 40]$ with `scipy.integrate.cumulative_trapezoid`. Verify with `np.allclose` that the numerical slope of the CDF (`np.gradient`) matches $f$ away from the ends, and find the median (the $x$ where the CDF equals $0.5$) with `np.interp`. Is the median smaller or larger than the mean, and what does that say about the shape of the curve?
6. **Report.** In 4 to 6 sentences, as if to a teammate who needs to normalise a similar curve: which rule you would use and why, how many strips were enough, what truncating the infinite range cost, and one thing that surprised you.

## What to hand in

A short notebook or script that runs top to bottom, with the plot, plus your written answers to tasks 2, 5, and 6.

## Stretch (optional)

1. Repeat task 2 on $[0, 2]$ instead of $[0, 3]$. The midpoint and trapezoid errors now fall much faster than the theory predicts. Look at the slope of $g$ at both ends of the interval and explain why.
2. Estimate $Z$ with Monte Carlo: average $g$ at $N$ uniform random points of $[0, 40]$ and multiply by $40$. Compare the error with Simpson's rule that uses the same number of function evaluations, and say which wins in one dimension and why that might change in many dimensions.
