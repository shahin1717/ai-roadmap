# Solutions: Limits and Continuity

Worked answers to the [problems](problems.md). The code outputs were produced by running the code. Numerical checks hold only up to rounding error, so the code compares with `np.isclose`, never `==`. Back to the [topic](../README.md).

## 1. A table of values

```python
import numpy as np
import sympy as sp

x = sp.symbols("x")
f = lambda t: (np.exp(t) - 1) / t
for t in [-0.1, -0.01, -0.001, 0.001, 0.01, 0.1]:
    print(f"x = {t:>6}   f(x) = {f(t):.6f}")
print("sympy:", sp.limit((sp.exp(x) - 1) / x, x, 0, "+-"))
```

```text
x =   -0.1   f(x) = 0.951626
x =  -0.01   f(x) = 0.995017
x = -0.001   f(x) = 0.999500
x =  0.001   f(x) = 1.000500
x =   0.01   f(x) = 1.005017
x =    0.1   f(x) = 1.051709
sympy: 1
```

The values close in on $1$ from both sides, so $\lim_{x\to0} f(x) = 1$. But $f(0)$ is $\frac00$, undefined: the limit exists even though the function has no value there (a hole). The next topic shows why this limit is $1$: it says the slope of $e^x$ at $0$ is $1$.

## 2. Sketch and decide

Here is one function with all the required properties: $f(x) = 2.5 - 0.5x$ for $x < 3$ except that $f(1) = 4$, $f(3) = 1$, and $f(x) = x$ for $x > 3$. The line through $(1, 2)$ and $(3, 1)$ gives the left pieces. A quick numerical check of the three values next to each point:

```python
def func(t):
    if np.isclose(t, 1, rtol=0, atol=1e-12):
        return 4.0
    if np.isclose(t, 3, rtol=0, atol=1e-12):
        return 1.0
    return 2.5 - 0.5 * t if t < 3 else t

h = 1e-7
for a in (1, 3):
    print(f"a = {a}:  just left = {func(a - h):.6f}   value = {func(a):.1f}   just right = {func(a + h):.6f}")
```

```text
a = 1:  just left = 2.000000   value = 4.0   just right = 2.000000
a = 3:  just left = 1.000000   value = 1.0   just right = 3.000000
```

- At $x = 1$ the limit ($2$) exists but differs from $f(1) = 4$: a **removable** discontinuity. Redefine $f(1) = 2$ and the break disappears.
- At $x = 3$ the one-sided limits are $1$ and $3$: a **jump** discontinuity. No single value of $f(3)$ can repair it (the value $f(3) = 1$ matches the left side only).

## 3. Algebra

(a) Factor and cancel: $\frac{(x-4)(x+4)}{x-4} = x + 4 \to 8$.

(b) Rationalize: $\frac{(x+25) - 25}{x(\sqrt{x+25}+5)} = \frac{1}{\sqrt{x+25}+5} \to \frac1{10}$.

(c) Expand: $\frac{9 + 6h + h^2 - 9}{h} = 6 + h \to 6$.

(d) Combine the top: $\frac1x - 1 = \frac{1-x}{x}$, so the fraction is $\frac{1-x}{x(x-1)} = \frac{-1}{x} \to -1$.

```python
h = sp.symbols("h")
print(sp.limit((x**2 - 16) / (x - 4), x, 4), sp.limit((sp.sqrt(x + 25) - 5) / x, x, 0),
      sp.limit(((3 + h)**2 - 9) / h, h, 0), sp.limit((1 / x - 1) / (x - 1), x, 1))
```

```text
8 1/10 6 -1
```

The limit in (c) is the slope of $y = x^2$ at $x = 3$, and (d) is the slope of $1/x$ at $1$, in disguise. You will meet both again as derivatives.

## 4. One-sided limits

(a) For $x > 2$, $|x-2| = x - 2$, so the fraction is $1$. For $x < 2$, $|x-2| = -(x-2)$, so it is $-1$. The one-sided limits are $-1$ (left) and $1$ (right), so the two-sided limit does not exist.

(b) For $x > 2$: $\frac{(x-2)(x+2)}{x-2} = x + 2 \to 4$. For $x < 2$: $\frac{(x-2)(x+2)}{-(x-2)} = -(x+2) \to -4$. The sides disagree, so the limit does not exist.

```python
a_expr = sp.Abs(x - 2) / (x - 2)
g_expr = (x**2 - 4) / sp.Abs(x - 2)
print("(a)", sp.limit(a_expr, x, 2, "-"), sp.limit(a_expr, x, 2, "+"))
print("(b)", sp.limit(g_expr, x, 2, "-"), sp.limit(g_expr, x, 2, "+"))
```

```text
(a) -1 1
(b) -4 4
```

## 5. The squeeze theorem

(a) Since $-1 \le \sin(1/x) \le 1$, multiplying by $x^2 \ge 0$ gives $-x^2 \le x^2\sin(1/x) \le x^2$. Both walls tend to $0$, so the limit is $0$.

(b) For $x > 0$, $-\frac1x \le \frac{\sin x}{x} \le \frac1x$. Both walls tend to $0$ as $x \to \infty$, so the limit is $0$.

```python
tt = np.linspace(0.001, 0.2, 1000)
print("(a) walls hold on a grid:", bool(np.all(np.abs(tt**2 * np.sin(1 / tt)) <= tt**2 + 1e-15)))
big = np.linspace(1, 1e4, 100000)
print("(b) walls hold on a grid:", bool(np.all(np.abs(np.sin(big) / big) <= 1 / big + 1e-15)))
print("sympy:", sp.limit(x**2 * sp.sin(1 / x), x, 0), sp.limit(sp.sin(x) / x, x, sp.oo))
```

```text
(a) walls hold on a grid: True
(b) walls hold on a grid: True
sympy: 0 0
```

## 6. A tolerance game

Here $|f(x) - 3| = |4x - 8| = 4|x - 2|$. This is below $\varepsilon$ when $|x - 2| < \varepsilon/4$, so $\delta = \varepsilon/4$ works. For $\varepsilon = 0.1$ that gives $\delta = 0.025$, and for $\varepsilon = 0.001$ it gives $\delta = 0.00025$. A $\delta$ of $\varepsilon/2$ is too generous: an $x$ just inside it, such as $x = 2 + 0.9\cdot\varepsilon/2$, has $|f(x) - 3| = 4 \cdot 0.45\,\varepsilon = 1.8\,\varepsilon$, outside the band. The function is steep (slope $4$), so the input band must be narrower than the output band.

```python
def wins(func, a, L, eps, delta, n=20001):
    xs = np.linspace(a - delta, a + delta, n)
    xs = xs[~np.isclose(xs, a, rtol=0, atol=1e-12)]
    xs = xs[np.abs(xs - a) < delta]
    return bool(np.all(np.abs(func(xs) - L) < eps))

line = lambda t: 4 * t - 5
for eps in [0.1, 0.001]:
    print(f"eps = {eps:<6} delta = eps/4 = {eps / 4:<8} wins: {wins(line, 2, 3, eps, eps / 4)}   delta = eps/2 wins: {wins(line, 2, 3, eps, eps / 2)}")
```

```text
eps = 0.1    delta = eps/4 = 0.025    wins: True   delta = eps/2 wins: False
eps = 0.001  delta = eps/4 = 0.00025  wins: True   delta = eps/2 wins: False
```

## 7. Fixing a function

(a) For $x \ne 3$, $\frac{x^2 - x - 6}{x - 3} = \frac{(x-3)(x+2)}{x-3} = x + 2$, so the limit at $3$ is $5$. Setting $f(3) = 5$ makes the function continuous.

(b) The left limit at $2$ is $2a + 1$, and $g(2) = 4 - a$ (the right-hand piece is continuous, so the right limit is also $4 - a$). Continuity needs $2a + 1 = 4 - a$, so $3a = 3$ and $a = 1$. Then both sides equal $3$.

```python
print("(a)", sp.limit((x**2 - x - 6) / (x - 3), x, 3))
a = sp.symbols("a")
solution = sp.solve(sp.Eq(2 * a + 1, 4 - a), a)
print("(b) a =", solution, "-> both sides at x = 2:", (2 * a + 1).subs(a, solution[0]), (4 - a).subs(a, solution[0]))
```

```text
(a) 5
(b) a = [1] -> both sides at x = 2: 3 3
```

## 8. Trapping a root

Let $p(x) = x^5 + x - 1$. It is a polynomial, so it is continuous. $p(0) = -1 < 0$ and $p(1) = 1 > 0$, so the intermediate value theorem gives a root in $[0, 1]$. Bisection by hand:

- $m = 0.5$: $p = 0.03125 + 0.5 - 1 = -0.46875 < 0$, so keep $[0.5, 1]$.
- $m = 0.75$: $p \approx 0.2373 + 0.75 - 1 = -0.0127 < 0$, so keep $[0.75, 1]$.
- $m = 0.875$: $p \approx 0.5129 + 0.875 - 1 = 0.3879 > 0$, so keep $[0.75, 0.875]$.

The width halves each step, so we need $2^{-n} \le 10^{-8}$, that is $n \ge \log_2(10^8) \approx 26.6$, so $27$ steps.

```python
p = lambda t: t**5 + t - 1
lo, hi = 0.0, 1.0
for step in range(1, 4):
    m = (lo + hi) / 2
    if p(lo) * p(m) <= 0:
        hi = m
    else:
        lo = m
    print(f"step {step}: m = {m}, p(m) = {p(m):+.5f}, bracket = [{lo}, {hi}]")
print("steps for 1e-8:", int(np.ceil(np.log2(1e8))))
print("sympy numeric root:", sp.nsolve(x**5 + x - 1, x, 0.8))
```

```text
step 1: m = 0.5, p(m) = -0.46875, bracket = [0.5, 1.0]
step 2: m = 0.75, p(m) = -0.01270, bracket = [0.75, 1.0]
step 3: m = 0.875, p(m) = +0.38791, bracket = [0.75, 0.875]
steps for 1e-8: 27
sympy numeric root: 0.754877666246693
```

## 9. Asymptotes and growth

(a) Divide top and bottom by $x^2$: $\frac{4 - 3/x + 1/x^2}{2 + 5/x^2} \to \frac42 = 2$. The horizontal asymptote is $y = 2$.

(b) Multiply by the conjugate: $\sqrt{x^2+x} - x = \frac{(x^2+x) - x^2}{\sqrt{x^2+x} + x} = \frac{x}{\sqrt{x^2+x}+x} = \frac{1}{\sqrt{1 + 1/x} + 1} \to \frac12$. The form looked like $\infty - \infty$ but is not zero.

(c) The bottom is $0$ at $x = 1$ and the top is $2$ there, so $x = 1$ is a vertical asymptote. From the right the bottom is a tiny positive number, so the limit is $+\infty$. From the left it is a tiny negative number, so the limit is $-\infty$.

(d) The table below shows $x^4$ ahead at $10$ and $15$, a tie at $16$ (both are $65536$), and $2^x$ ahead from $17$ on. Exponential growth beats any power eventually.

```python
print("(a)", sp.limit((4*x**2 - 3*x + 1) / (2*x**2 + 5), x, sp.oo))
print("(b)", sp.limit(sp.sqrt(x**2 + x) - x, x, sp.oo))
print("(c)", sp.limit((x**2 + 1) / (x - 1), x, 1, "-"), sp.limit((x**2 + 1) / (x - 1), x, 1, "+"))
print("(d)    x        x^4          2^x")
for xv in [10, 15, 16, 17, 30]:
    print(f"    {xv:>3} {xv**4:>10} {2**xv:>12}")
```

```text
(a) 2
(b) 1/2
(c) -oo oo
(d)    x        x^4          2^x
     10      10000         1024
     15      50625        32768
     16      65536        65536
     17      83521       131072
     30     810000   1073741824
```

## 10. Bisection from scratch

```python
from scipy.optimize import brentq

def bisect(func, a, b, tol=1e-6):
    if func(a) * func(b) > 0:
        raise ValueError("func(a) and func(b) must have opposite signs")
    steps = 0
    while b - a > tol:
        m = (a + b) / 2
        if func(a) * func(m) <= 0:
            b = m
        else:
            a = m
        steps += 1
    return (a + b) / 2, steps

residual = lambda t: np.cos(t) - t          # cos x = x  <=>  residual(x) = 0
for tol in [1e-3, 1e-6, 1e-10]:
    root, steps = bisect(residual, 0, 1, tol)
    print(f"tol = {tol:.0e}: root = {root:.10f}, steps = {steps}, log2(1/tol) = {np.log2(1 / tol):.1f}")
print("agrees with brentq:", np.isclose(root, brentq(residual, 0, 1), atol=1e-9))

try:
    bisect(lambda t: t**2 + 1, -1, 1)
except ValueError as err:
    print("error:", err)
```

```text
tol = 1e-03: root = 0.7387695312, steps = 10, log2(1/tol) = 10.0
tol = 1e-06: root = 0.7390847206, steps = 20, log2(1/tol) = 19.9
tol = 1e-10: root = 0.7390851332, steps = 34, log2(1/tol) = 33.2
agrees with brentq: True
error: func(a) and func(b) must have opposite signs
```

(a) The number of steps matches $\lceil\log_2(1/\text{tol})\rceil$ almost exactly: every extra factor of $1000$ in accuracy costs about $10$ more steps. (b) The root is $0.7390851\ldots$, and it agrees with the library solver. (c) For $x^2 + 1$ there is no sign change on $[-1, 1]$ (it is always positive, and has no root), so the function refuses to start instead of returning a meaningless answer.
