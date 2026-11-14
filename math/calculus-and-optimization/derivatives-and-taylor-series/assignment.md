# Assignment: Build Your Own Autodiff

Deep-learning libraries compute derivatives automatically. In this assignment you build the smallest working version, a forward-mode automatic differentiation class, use it on a composite function and on a tiny network, check it against SymPy and finite differences, and then use Taylor polynomials to see how the error of a polynomial approximation falls with its degree. It takes about 60-90 minutes and needs only NumPy, SymPy, and Matplotlib. Back to the [topic](README.md).

## The setup

The functions and numbers are invented for this exercise.

```python
import math
import numpy as np
import sympy as sp

x0 = 1.3                                    # where we differentiate the composite function
# composite: g(x) = ln(1 + x^2) * e^(-x/2)

x_in, target = 0.7, 1.0                     # one input and its target
w1, w2, b = 1.2, 0.9, -0.3                  # weights of the tiny network
# network: y = w2 * sigmoid(w1 * x_in + b),   loss L = (y - target)^2
```

## Tasks

1. **Build the class.** Write a class `Dual` that stores a value `val` and a derivative `der`. Support `+`, `-`, `*`, `/`, and `**` with a number exponent, with a plain number allowed on either side (`2 * u`, `1 + u`, `3 - u`, `1 / u`), plus unary minus. Then write a helper `apply(fun, dfun, u)` that applies the chain rule, and use it to define `exp`, `log`, and `sigmoid` for dual numbers. The whole thing should be about 30 lines. Test it on $u^2$ at $u = 3$ (the derivative should be $6$).
2. **A composite function.** Compute $g'(x_0)$ for the $g$ above with your class (seed the input's derivative with $1$). Compare with SymPy's exact derivative and with a centered finite difference with $h = 10^{-5}$, and confirm both comparisons with `np.isclose`. Print all three numbers.
3. **The tiny network.** Differentiate the loss $L = (y - t)^2$ with respect to $w_1$, $w_2$, and $b$ using three passes of your class (seed one parameter at a time). Also compute the three derivatives by hand with the chain rule (write the formulas in a comment) and with SymPy. Print a small table of the three methods and check agreement with `np.allclose`.
4. **No step size needed.** For $g$ at $x_0$, compute the absolute error of the centered finite difference for $h = 10^{-1}, 10^{-2}, \dots, 10^{-14}$, measured against SymPy's value, and the absolute error of your autodiff. Plot or print them. In three or four sentences, explain why the finite-difference error first falls and then rises, and why autodiff has no such trade-off.
5. **Taylor error against degree.** Let $f(x) = \ln(1+x)$. For $x = 0.4$, $x = 0.9$, and $x = 1.5$, use `sp.series(f, x, 0, n + 1).removeO()` to get the Maclaurin polynomial $T_n$ for $n = 1, 2, 3, 4, 6, 8, 12$ and compute the absolute error $|f(x) - T_n(x)|$. Plot error against degree on a log axis (one line per $x$). Describe what happens at each of the three points and connect it to the radius of convergence.
6. **Error against distance.** For the fixed degree $n = 3$, compute the error of $T_3$ at $20$ values of $x$ between $0.01$ and $0.1$ and fit a line to $\log(\text{error})$ against $\log x$ with `np.polyfit`. The slope should be close to $n + 1 = 4$. Explain in one sentence why.
7. **Report.** In 5-8 sentences, as if to a teammate: what forward mode needs for a function of many inputs, why libraries use reverse mode for training, which approach (symbolic, numeric, automatic) you would use to check a hand-written gradient and why, and one thing that surprised you.

## What to hand in

A short notebook or script that runs top to bottom, with the plots, plus your written explanations for tasks 4, 5, 6, and 7.

## Stretch (optional)

Add `sin`, `cos`, and `relu` to your class and differentiate $\mathrm{relu}(w_1 x + b)\,w_2$ at a point where $w_1 x + b < 0$. What do you get, and what does it say about a dead ReLU? Then add a `__repr__` so that `print(u)` shows both the value and the derivative.
