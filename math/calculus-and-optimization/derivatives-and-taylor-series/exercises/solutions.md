# Solutions: Derivatives and Taylor Series

Worked answers to the [problems](problems.md). The code outputs were produced by running the code, and SymPy shows some answers in a different but equal form. Numerical checks hold only up to rounding error, so the code compares with `np.isclose` and `np.allclose`, never `==`. Back to the [topic](../README.md).

## 1. From the definition

(a) $f(2+h) - f(2) = 3(4 + 4h + h^2) - (2 + h) - 10 = 11h + 3h^2$, so the quotient is $11 + 3h \to 11$. With $f(2) = 10$ the tangent line is $y = 10 + 11(x - 2)$.

(b) Multiply by the conjugate: $\dfrac{\sqrt{4+h} - 2}{h} = \dfrac{(4+h) - 4}{h\,(\sqrt{4+h} + 2)} = \dfrac{1}{\sqrt{4+h} + 2} \to \dfrac14$. With $f(4) = 2$ the tangent line is $y = 2 + \frac14(x - 4)$.

```python
import numpy as np
import sympy as sp

x, h, a = sp.symbols("x h a")
for name, expr, pt in [("3x^2 - x", 3*x**2 - x, 2), ("sqrt(x)", sp.sqrt(x), 4)]:
    quotient = (expr.subs(x, pt + h) - expr.subs(x, pt)) / h
    print(f"f = {name}: f'({pt}) = {sp.limit(quotient, h, 0)},  f({pt}) = {expr.subs(x, pt)}")
```

```text
f = 3x^2 - x: f'(2) = 11,  f(2) = 10
f = sqrt(x): f'(4) = 1/4,  f(4) = 2
```

## 2. The rules

(a) Power and sum: $10x^4 - \dfrac{3}{2\sqrt{x}} - \dfrac{4}{x^2}$.  (b) Product: $2x\ln x + x^2\cdot\frac1x = 2x\ln x + x$.  (c) Quotient: $\dfrac{1\cdot(x-1) - (x+1)\cdot 1}{(x-1)^2} = \dfrac{-2}{(x-1)^2}$.  (d) Product: $e^x\cos x - e^x\sin x = e^x(\cos x - \sin x)$.

```python
exprs = {"(a)": 2*x**5 - 3*sp.sqrt(x) + 4/x, "(b)": x**2 * sp.log(x),
         "(c)": (x + 1) / (x - 1), "(d)": sp.exp(x) * sp.cos(x)}
for label, e in exprs.items():
    print(label, sp.diff(e, x))
```

```text
(a) 10*x**4 - 4/x**2 - 3/(2*sqrt(x))
(b) 2*x*log(x) + x
(c) 1/(x - 1) - (x + 1)/(x - 1)**2
(d) -exp(x)*sin(x) + exp(x)*cos(x)
```

SymPy leaves (c) as $\frac{1}{x-1} - \frac{x+1}{(x-1)^2}$, which is the same thing over a common denominator.

## 3. The chain rule

(a) $7(2x-5)^6\cdot 2 = 14(2x-5)^6$.  (b) $\cos(3x^2)\cdot 6x$.  (c) $e^{-x^2/2}\cdot(-x)$.  (d) $\dfrac{1}{1+x^2}\cdot 2x$.  (e) The nesting is $u = \sin x$, $v = 1 + u^2$, $y = \sqrt{v}$, so $\dfrac{dy}{dx} = \dfrac{1}{2\sqrt v}\cdot 2u\cdot\cos x = \dfrac{\sin x\cos x}{\sqrt{1 + \sin^2 x}}$.

```python
exprs = {"(a)": (2*x - 5)**7, "(b)": sp.sin(3*x**2), "(c)": sp.exp(-x**2 / 2),
         "(d)": sp.log(1 + x**2), "(e)": sp.sqrt(1 + sp.sin(x)**2)}
for label, e in exprs.items():
    print(label, sp.diff(e, x))
```

```text
(a) 14*(2*x - 5)**6
(b) 6*x*cos(3*x**2)
(c) -x*exp(-x**2/2)
(d) 2*x/(x**2 + 1)
(e) sin(x)*cos(x)/sqrt(sin(x)**2 + 1)
```

## 4. Sigmoid and tanh

(a) Write $\sigma = (1 + e^{-x})^{-1}$. The outside is $u^{-1}$ with derivative $-u^{-2}$ and the inside is $u = 1 + e^{-x}$ with derivative $-e^{-x}$, so $\sigma' = \dfrac{e^{-x}}{(1+e^{-x})^2}$. Since $\sigma = \frac{1}{1+e^{-x}}$ and $1 - \sigma = \frac{e^{-x}}{1+e^{-x}}$, this is $\sigma(1-\sigma)$. At $0$ it is $\frac12\cdot\frac12 = 0.25$, and at $\pm 3$ it is about $0.0452$ (the slope is an even function of $x$).

(b) $\frac{d}{dx}\big[2\sigma(2x) - 1\big] = 2\cdot\sigma'(2x)\cdot 2 = 4\,\sigma(2x)\big(1 - \sigma(2x)\big)$. Let $t = \tanh x = 2\sigma(2x) - 1$. Then $\sigma(2x) = \frac{1+t}{2}$ and $1 - \sigma(2x) = \frac{1-t}{2}$, so the derivative is $4\cdot\frac{1+t}{2}\cdot\frac{1-t}{2} = 1 - t^2$.

(c) Each slope is at most $\frac14$, so the product is at most $\left(\frac14\right)^{10} \approx 9.5\times10^{-7}$.

```python
sigma = lambda z: 1 / (1 + np.exp(-z))
dsigma = lambda z: sigma(z) * (1 - sigma(z))
print("sigma' at 0, 3, -3:", [round(float(dsigma(z)), 6) for z in (0.0, 3.0, -3.0)])
xs = np.linspace(-10, 10, 2001)
print("tanh' = 1 - tanh^2 check:", np.allclose(np.gradient(np.tanh(xs), xs), 1 - np.tanh(xs)**2, atol=1e-4),
      "| tanh = 2 sigma(2x) - 1:", np.allclose(np.tanh(xs), 2 * sigma(2 * xs) - 1))
print("max of sigma' on a grid:", dsigma(xs).max(), "  4^-10 =", 0.25**10)
```

```text
sigma' at 0, 3, -3: [0.25, 0.045177, 0.045177]
tanh' = 1 - tanh^2 check: True | tanh = 2 sigma(2x) - 1: True
max of sigma' on a grid: 0.25   4^-10 = 9.5367431640625e-07
```

## 5. Where the derivative fails, and gluing pieces

(a) $|x^2 - 4|$ equals $x^2 - 4$ where that is positive and $4 - x^2$ where it is negative, so it has corners at $x = \pm 2$, where the inside is $0$. The slopes there are $-4$ on the left and $4$ on the right at both points, so there is no derivative. Everywhere else the function is smooth.

(b) Continuity at $1$: $a\cdot1 + b = 1^2$, so $a + b = 1$. Equal slopes: the left slope is $2\cdot1 = 2$ and the right slope is $a$, so $a = 2$ and then $b = -1$.

(c) The slope is $0.1$ on the left and $1$ on the right, so Leaky ReLU has a corner at $0$ and no derivative there, like ReLU. (In training the library picks a value at $0$.)

```python
f5 = sp.Abs(x**2 - 4)
inner = x**2 - 4
for pt, left_sign, right_sign in [(2, -1, 1), (-2, 1, -1)]:      # |inner| = sign * inner on each side
    print(f"at x = {pt}: left slope {left_sign * sp.diff(inner, x).subs(x, pt)}, right slope {right_sign * sp.diff(inner, x).subs(x, pt)}")
A, B = sp.symbols("A B")
sol = sp.solve([sp.Eq(1, A + B), sp.Eq(2, A)], [A, B])
print("continuity and equal slopes at 1:", sol)
```

```text
at x = 2: left slope -4, right slope 4
at x = -2: left slope -4, right slope 4
continuity and equal slopes at 1: {A: 2, B: -1}
```

## 6. Tangent line and approximation error

(a) $f(2) = 8 - 4 = 4$ and $f'(x) = 3x^2 - 2$, so $f'(2) = 10$. The tangent line is $y = 4 + 10(x-2)$.

(b) For $\ln x$ at $1$: $f(1) = 0$ and $f'(1) = 1$, so $L(x) = x - 1$, and for $x = 1 + d$ the estimate is $d$. The errors are about $1.2\times10^{-3}$ for $d = 0.05$ and $1.2\times10^{-5}$ for $d = 0.005$. Ten times closer gave about a hundred times smaller error (the ratio is $97$), because the error is about $\frac{d^2}{2}$ (the next Taylor term, $-\frac{d^2}{2}$, is the one the line leaves out). For $d = 0.0005$ it predicts about $1.25\times10^{-7}$.

```python
y6 = x**3 - 2*x
print("f(2) =", y6.subs(x, 2), " f'(2) =", sp.diff(y6, x).subs(x, 2))
errs = []
for d in [0.05, 0.005]:
    est, true = d, np.log(1 + d)
    errs.append(abs(est - true))
    print(f"ln(1 + {d}) estimate {est:.4f}, true {true:.6f}, error {errs[-1]:.3e}, error/d^2 = {errs[-1] / d**2:.4f}")
print("error ratio:", errs[0] / errs[1])
```

```text
f(2) = 4  f'(2) = 10
ln(1 + 0.05) estimate 0.0500, true 0.048790, error 1.210e-03, error/d^2 = 0.4839
ln(1 + 0.005) estimate 0.0050, true 0.004988, error 1.246e-05, error/d^2 = 0.4983
error ratio: 97.10935526387631
```

## 7. Critical points and a one-variable fit

(a) $f'(x) = 4x^3 - 16x = 4x(x-2)(x+2)$, so the critical points are $x = -2, 0, 2$. The sign of $f'$ runs $-,+,-,+$ across them: $x = \pm 2$ are local minima ($f = -13$) and $x = 0$ is a local maximum ($f = 3$). The second derivative $f''(x) = 12x^2 - 16$ agrees: $f''(\pm2) = 32 > 0$ and $f''(0) = -16 < 0$. The inflection points are where $f'' = 0$: $x = \pm\frac{2}{\sqrt3} \approx \pm1.155$. Since $f \to \infty$ at both ends, the global minimum is $-13$, reached at $x = -2$ and at $x = 2$.

(b) $L'(w) = 2\big[(w-3) + 2(2w-5) + 4(4w-9)\big] = 2(21w - 49) = 42w - 98$, which is zero at $w = \frac{98}{42} = \frac73$. $L''(w) = 42 > 0$, so it is a minimum.

```python
f7 = x**4 - 8*x**2 + 3
d1, d2 = sp.diff(f7, x), sp.diff(f7, x, 2)
print("f' =", sp.factor(d1), "| critical points:", sp.solve(d1, x), "| f'' =", d2)
for c in sp.solve(d1, x):
    sg = lambda v: "+" if v > 0 else "-"
    print(f"x = {c}: f = {f7.subs(x, c)}, f'' = {d2.subs(x, c)}, sign of f' on the left/right: {sg(d1.subs(x, c - 0.1))} {sg(d1.subs(x, c + 0.1))}")
print("inflection points:", sp.solve(d2, x))
xi, yi = np.array([1.0, 2.0, 4.0]), np.array([3.0, 5.0, 9.0])
w = sp.symbols("w")
L = sum((w * p - q)**2 for p, q in zip(xi, yi))
print("L'(w) =", sp.expand(sp.diff(L, w)), "| w =", sp.solve(sp.diff(L, w), w)[0], "= 7/3:", np.isclose(float(sp.solve(sp.diff(L, w), w)[0]), 7 / 3), "| L'' =", sp.diff(L, w, 2))
```

```text
f' = 4*x*(x - 2)*(x + 2) | critical points: [-2, 0, 2] | f'' = 4*(3*x**2 - 4)
x = -2: f = -13, f'' = 32, sign of f' on the left/right: - +
x = 0: f = 3, f'' = -16, sign of f' on the left/right: + -
x = 2: f = -13, f'' = 32, sign of f' on the left/right: - +
inflection points: [-2*sqrt(3)/3, 2*sqrt(3)/3]
L'(w) = 42.0*w - 98.0 | w = 2.33333333333333 = 7/3: True | L'' = 42.0000000000000
```

## 8. Taylor polynomials

(a) The derivatives of $\sin$ at $0$ are $0, 1, 0, -1$, so $T_3(x) = x - \frac{x^3}{6}$ and $T_3(0.3) = 0.3 - 0.0045 = 0.2955$. The true value is $0.295520\ldots$, an error of about $2.02\times10^{-5}$, very close to the first omitted term $\frac{0.3^5}{120} = 2.025\times10^{-5}$.

(b) $f(4) = 2$, $f'(4) = \frac14$, and $f''(x) = -\frac14 x^{-3/2}$ gives $f''(4) = -\frac{1}{32}$. So $T_2(x) = 2 + \frac{x-4}{4} - \frac{(x-4)^2}{64}$. At $x = 4.4$ the linear estimate is $2.1$ and the quadratic estimate is $2.1 - 0.0025 = 2.0975$, against the true value $2.09762$: the error drops from about $2.4\times10^{-3}$ to $1.2\times10^{-4}$.

```python
T3 = x - x**3 / 6
print("T3(0.3) =", float(T3.subs(x, 0.3)), " sin(0.3) =", np.sin(0.3), " error =", abs(np.sin(0.3) - float(T3.subs(x, 0.3))), " next term =", 0.3**5 / 120)
g = sp.sqrt(x)
T2 = sum(sp.diff(g, x, k).subs(x, 4) / sp.factorial(k) * (x - 4)**k for k in range(3))
print("T2 =", sp.expand(T2))
print("sqrt(4.4): linear", 2 + 0.4 / 4, " quadratic", float(T2.subs(x, 4.4)), " true", np.sqrt(4.4))
```

```text
T3(0.3) = 0.2955  sin(0.3) = 0.29552020666133955  error = 2.0206661339561283e-05  next term = 2.0249999999999994e-05
T2 = -x**2/64 + 3*x/8 + 3/4
sqrt(4.4): linear 2.1  quadratic 2.0975  true 2.0976176963403033
```

## 9. Series and radius of convergence

(a) Replace $x$ by $-x^2$ in $\frac{1}{1-x} = 1 + x + x^2 + \dots$ to get $\frac{1}{1+x^2} = 1 - x^2 + x^4 - x^6 + \dots$. This converges for $|{-x^2}| < 1$, that is $|x| < 1$, so $R = 1$ (even though the real graph has no break, the function misbehaves at the complex numbers $\pm i$, at distance $1$ from $0$). At $x = 0.5$ the partial sum is close to the function value $0.8$. At $x = 1.5$, outside the radius, the partial sum is far off and keeps growing with more terms.

(b) Replace $u$ by $-x^2$ in $e^u = 1 + u + \frac{u^2}{2} + \dots$: $e^{-x^2} = 1 - x^2 + \frac{x^4}{2} - \frac{x^6}{6} + \dots$. The series of $e^u$ converges for every $u$, so this one converges for every $x$ and $R = \infty$.

```python
print(sp.series(1 / (1 + x**2), x, 0, 8))
print(sp.series(sp.exp(-x**2), x, 0, 7))
for t in (0.5, 1.5):
    partial = sum((-1)**k * t**(2 * k) for k in range(6))
    print(f"x = {t}: 6-term partial sum = {partial:.4f}, true value = {1 / (1 + t**2):.4f}")
```

```text
1 - x**2 + x**4 - x**6 + O(x**8)
1 - x**2 + x**4/2 - x**6/6 + O(x**7)
x = 0.5: 6-term partial sum = 0.7998, true value = 0.8000
x = 1.5: 6-term partial sum = -39.6143, true value = 0.3077
```

## 10. Checking derivatives in code

(a) The centered finite difference agrees with $\sigma(1-\sigma)$ at all five points. The slope first falls below $0.01$ near $x = 4.58$ (since $\sigma'(x) \approx e^{-x}$ for large $x$, this is close to $\ln 100 \approx 4.6$).

(b) The two errors are about $6.9\times10^{-5}$ and $6.7\times10^{-9}$, with ratio about $10^4$. The rule says the error of a degree-3 polynomial scales like $|x|^4$, so a tenfold smaller $x$ should give $10^4$ times smaller error. (The ratio is a little above $10^4$ because the next term, $\frac{x^5}{120}$, also contributes more at the larger $x$.)

(c) Three jumps give $0.69404$, and five give $0.693147$, which `np.isclose` accepts as $\ln 2$.

```python
z = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
eps = 1e-5
fd = (sigma(z + eps) - sigma(z - eps)) / (2 * eps)
print("finite difference matches sigma(1 - sigma):", np.allclose(fd, dsigma(z)))
grid = np.linspace(0, 10, 100001)
print("first x with sigma' < 0.01:", grid[np.argmax(dsigma(grid) < 0.01)])
from math import factorial
T3e = lambda t: sum(t**k / factorial(k) for k in range(4))
e1, e2 = abs(np.exp(0.2) - T3e(0.2)), abs(np.exp(0.02) - T3e(0.02))
print(f"errors {e1:.3e} and {e2:.3e}, ratio {e1 / e2:.0f}  (rule predicts 10^4 = {10**4})")
fp, fpp = (lambda s: np.exp(s) - 2), np.exp
s, path = 0.0, []
for _ in range(5):
    s = s - fp(s) / fpp(s)
    path.append(s)
print("after 3 jumps:", path[2], " after 5 jumps:", path[4], " ln 2 =", np.log(2), " close:", np.isclose(path[4], np.log(2)))
```

```text
finite difference matches sigma(1 - sigma): True
first x with sigma' < 0.01: 4.5849
errors 6.942e-05 and 6.693e-09, ratio 10372  (rule predicts 10^4 = 10000)
after 3 jumps: 0.6940422999189153  after 5 jumps: 0.6931471805600254  ln 2 = 0.6931471805599453  close: True
```
