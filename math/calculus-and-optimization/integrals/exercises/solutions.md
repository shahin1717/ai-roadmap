# Solutions: Integrals

Worked answers to the [problems](problems.md). The code outputs were produced by running the code. Numerical checks hold only up to rounding error, so the code compares with `np.isclose`, never `==`. Back to the [topic](../README.md).

## 1. A Riemann sum by hand

(a) The strip width is $\Delta x = 2/4 = 0.5$ and the edges are $0, 0.5, 1, 1.5, 2$. Left sum: $0.5\,(0 + 0.25 + 1 + 2.25) = 1.75$. Right sum: $0.5\,(0.25 + 1 + 2.25 + 4) = 3.75$.

(b) $\int_0^2 x^2\,dx = \left[\frac{x^3}{3}\right]_0^2 = \frac83 \approx 2.667$, which lies between $1.75$ and $3.75$.

(c) Each strip has width $2/n$ and $f$ rises by $f(2) - f(0) = 4$ in total, so the gap is $\frac{2}{n}\cdot 4 = \frac8n$. We need $\frac8n < 0.01$, so $n > 800$: $n = 801$ strips work, while $n = 800$ gives a gap of exactly $0.01$.

```python
import numpy as np
import sympy as sp

f = lambda x: x**2
n, a, b = 4, 0, 2
dx = (b - a) / n
edges = a + dx * np.arange(n + 1)
left_sum = np.sum(f(edges[:-1])) * dx
right_sum = np.sum(f(edges[1:])) * dx
x = sp.symbols("x")
exact = sp.integrate(x**2, (x, 0, 2))
print("left:", left_sum, "right:", right_sum, "exact:", exact, float(exact))
print("exact between the sums:", left_sum < float(exact) < right_sum)
print("gap for n strips:", (b - a) / 800 * (f(2) - f(0)), "at n = 800;", (b - a) / 801 * (f(2) - f(0)), "at n = 801")
```

```text
left: 1.75 right: 3.75 exact: 8/3 2.6666666666666665
exact between the sums: True
gap for n strips: 0.01 at n = 800; 0.009987515605493134 at n = 801
```

## 2. Antiderivatives

(a) Term by term with the table: $\int 4x^3\,dx = x^4$, $\int -\frac3x\,dx = -3\ln\lvert x\rvert$, $\int 2e^x\,dx = 2e^x$, $\int -\sin x\,dx = \cos x$. So the answer is $x^4 - 3\ln\lvert x\rvert + 2e^x + \cos x + C$ (SymPy writes `log(x)` and drops the $C$).

(b) $f(x) = \sin x + 3x^2 + C$, and $f(0) = 0 + 0 + C = 2$ gives $C = 2$, so $f(x) = \sin x + 3x^2 + 2$.

```python
import sympy as sp
x = sp.symbols("x", positive=True)
integrand = 4*x**3 - 3/x + 2*sp.exp(x) - sp.sin(x)
F = sp.integrate(integrand, x)
print("(a) antiderivative:", F, "| derivative minus integrand:", sp.simplify(sp.diff(F, x) - integrand))
f = sp.integrate(sp.cos(x) + 6*x, x)           # no +C from SymPy: add it back
C = sp.symbols("C")
sol = sp.solve(sp.Eq((f + C).subs(x, 0), 2), C)[0]
print("(b) f(x) =", f + sol)
```

```text
(a) antiderivative: x**4 + 2*exp(x) - 3*log(x) + cos(x) | derivative minus integrand: 0
(b) f(x) = 3*x**2 + sin(x) + 2
```

## 3. Both parts

(a) By Part 1 of the fundamental theorem, the derivative of an area function with respect to its upper limit is the integrand evaluated there: $e^{-x^2}$. (There is no elementary antiderivative, and none is needed.)

(b) Let $A(u) = \int_0^u \cos t\,dt$. The expression is $A(x^2)$, so the chain rule gives $A'(x^2)\cdot 2x = \cos(x^2)\cdot 2x$. Check: the integral is $\sin(x^2)$, whose derivative is the same.

(c) $\int_1^4 \sqrt x\,dx = \left[\frac23 x^{3/2}\right]_1^4 = \frac23(8 - 1) = \frac{14}{3}$. And $\int_0^3 (x-1)\,dx = \left[\frac{x^2}{2} - x\right]_0^3 = \frac92 - 3 = \frac32$.

(d) The graph of $x - 1$ is below the axis on $[0, 1]$ (a triangle of area $\frac12$) and above it on $[1, 3]$ (a triangle of area $2$). The signed integral is $-\frac12 + 2 = \frac32$, while the unsigned area is $\frac12 + 2 = \frac52$. The part below the axis counts negative in the integral and positive in the total area.

```python
import numpy as np
import sympy as sp
from scipy.integrate import quad
x, t = sp.symbols("x t", positive=True)
# (a) and (b): numerical check of the derivative of the area function at x = 0.7
h = 1e-6
A = lambda u: quad(lambda s: np.exp(-s**2), 0, u)[0]
print("(a) numeric slope:", (A(0.7 + h) - A(0.7 - h)) / (2 * h), "| e^(-x^2) at 0.7:", np.exp(-0.7**2))
B = lambda u: quad(np.cos, 0, u**2)[0]
print("(b) numeric slope:", (B(0.7 + h) - B(0.7 - h)) / (2 * h), "| 2x cos(x^2) at 0.7:", 2 * 0.7 * np.cos(0.7**2))
print("(b) SymPy:", sp.diff(sp.integrate(sp.cos(t), (t, 0, x**2)), x))
print("(c)", sp.integrate(sp.sqrt(x), (x, 1, 4)), sp.integrate(x - 1, (x, 0, 3)))
unsigned = sp.integrate(1 - x, (x, 0, 1)) + sp.integrate(x - 1, (x, 1, 3))
print("(d) unsigned area:", unsigned, "| numeric:", quad(lambda s: abs(s - 1), 0, 3)[0])
```

```text
(a) numeric slope: 0.6126263941697019 | e^(-x^2) at 0.7: 0.6126263941844161
(b) numeric slope: 1.2352660020698014 | 2x cos(x^2) at 0.7: 1.2352660020541701
(b) SymPy: 2*x*cos(x**2)
(c) 14/3 3/2
(d) unsigned area: 5/2 | numeric: 2.5
```

## 4. Area between curves

The curves cross where $x^2 = 8 - x^2$, that is $x^2 = 4$, at $x = -2$ and $x = 2$. At $x = 0$ the curve $8 - x^2$ is at $8$ and $x^2$ is at $0$, so $8 - x^2$ is on top between the crossings:

$$\int_{-2}^{2}\big((8 - x^2) - x^2\big)\,dx = \int_{-2}^{2}(8 - 2x^2)\,dx = \left[8x - \frac{2x^3}{3}\right]_{-2}^{2} = \left(16 - \frac{16}{3}\right)\cdot 2 = \frac{64}{3}.$$

```python
import sympy as sp
x = sp.symbols("x")
crossings = sp.solve(sp.Eq(x**2, 8 - x**2), x)
area = sp.integrate((8 - x**2) - x**2, (x, crossings[0], crossings[1]))
print("crossings:", crossings, "| area:", area, float(area))
print("top curve at x = 0 is 8 - x^2:", (8 - x**2).subs(x, 0) > (x**2).subs(x, 0))
```

```text
crossings: [-2, 2] | area: 64/3 21.333333333333332
top curve at x = 0 is 8 - x^2: True
```

## 5. Average value and a volume

(a) The area is $\int_0^2 3x^2\,dx = [x^3]_0^2 = 8$, so the average value is $\frac{8}{2} = 4$. Solving $3c^2 = 4$ gives $c = \frac{2}{\sqrt3} \approx 1.155$ (the positive root, which lies in $[0, 2]$).

(b) $V = \pi\int_1^4 \left(\frac1x\right)^2dx = \pi\left[-\frac1x\right]_1^4 = \pi\left(1 - \frac14\right) = \frac{3\pi}{4} \approx 2.356$. The sum of many thin discs agrees.

```python
import numpy as np
import sympy as sp
x = sp.symbols("x", positive=True)
avg = sp.integrate(3 * x**2, (x, 0, 2)) / 2
c = sp.solve(sp.Eq(3 * x**2, avg), x)
print("(a) average:", avg, "| c =", c, [float(v) for v in c])
V = sp.integrate(sp.pi * (1 / x)**2, (x, 1, 4))
print("(b) volume:", V, float(V))
n = 100_000; dx = 3 / n; mids = 1 + dx * (np.arange(n) + 0.5)
print("sum of discs:", np.sum(np.pi * (1 / mids)**2 * dx), "| close:", np.isclose(np.sum(np.pi * (1 / mids)**2 * dx), float(V)))
```

```text
(a) average: 4 | c = [2*sqrt(3)/3] [1.1547005383792515]
(b) volume: 3*pi/4 2.356194490192345
sum of discs: 2.356194489960407 | close: True
```

## 6. Substitution

(a) Let $u = x^3$, so $du = 3x^2\,dx$ and the integral is $\int e^u\,du = e^{x^3} + C$.

(b) Let $u = \sin x$, so $du = \cos x\,dx$ and the integral is $\int u^3\,du = \frac{u^4}{4} + C = \frac{\sin^4 x}{4} + C$.

(c) Let $u = x^2 + 1$, so $x\,dx = \frac{du}{2}$. The limits become $u = 1$ (at $x = 0$) and $u = 2$ (at $x = 1$): $\frac12\int_1^2 u^{-2}\,du = \frac12\left[-\frac1u\right]_1^2 = \frac12\left(1 - \frac12\right) = \frac14$.

```python
import sympy as sp
x = sp.symbols("x")
for name, integrand, F in [("(a)", 3*x**2*sp.exp(x**3), sp.exp(x**3)),
                           ("(b)", sp.cos(x)*sp.sin(x)**3, sp.sin(x)**4 / 4)]:
    print(name, "derivative minus integrand:", sp.simplify(sp.diff(F, x) - integrand))
u = sp.symbols("u", positive=True)
print("(c) in x:", sp.integrate(x / (x**2 + 1)**2, (x, 0, 1)), "| in u, limits 1 to 2:", sp.integrate(sp.Rational(1, 2) * u**-2, (u, 1, 2)))
```

```text
(a) derivative minus integrand: 0
(b) derivative minus integrand: 0
(c) in x: 1/4 | in u, limits 1 to 2: 1/4
```

## 7. Integration by parts

(a) First pass: $u = x^2$, $dv = e^x\,dx$, so $du = 2x\,dx$ and $v = e^x$: $\int x^2e^x\,dx = x^2e^x - 2\int xe^x\,dx$. The remaining integral is the README example, $\int xe^x\,dx = (x-1)e^x$ (a second pass with $u = x$). So the answer is $x^2e^x - 2(x-1)e^x + C = (x^2 - 2x + 2)e^x + C$.

(b) With $u = x$, $dv = \sin x\,dx$, $v = -\cos x$: $\int x\sin x\,dx = -x\cos x + \int\cos x\,dx = -x\cos x + \sin x$. Then $[-x\cos x + \sin x]_0^\pi = (\pi + 0) - (0 + 0) = \pi$.

```python
import sympy as sp
x = sp.symbols("x")
integrand = x**2 * sp.exp(x)
F = sp.exp(x) * (x**2 - 2*x + 2)
print("(a) derivative minus integrand:", sp.simplify(sp.diff(F, x) - integrand), "| SymPy:", sp.simplify(sp.integrate(integrand, x)))
print("(b)", sp.integrate(x * sp.sin(x), (x, 0, sp.pi)), "| antiderivative:", sp.integrate(x * sp.sin(x), x))
```

```text
(a) derivative minus integrand: 0 | SymPy: (x**2 - 2*x + 2)*exp(x)
(b) pi | antiderivative: -x*cos(x) + sin(x)
```

## 8. Improper integrals

(a) $p = 3 > 1$, so it converges: $\frac{1}{p - 1} = \frac12$.

(b) $\int_0^b 3e^{-3x}\,dx = 1 - e^{-3b} \to 1$. It converges to $1$ (this is the exponential density with $\lambda = 3$).

(c) $p = \frac12 < 1$, so it diverges. Directly, $\int_1^b x^{-1/2}\,dx = 2\sqrt b - 2$, which grows without bound (about $1998$ already at $b = 10^6$).

(d) By parts, $\int xe^{-x}\,dx = -xe^{-x} - e^{-x}$, so $\int_0^b xe^{-x}\,dx = 1 - (b + 1)e^{-b}$. Since $(b+1)e^{-b} \to 0$ (the exponential beats every power), the value is $1$.

```python
import numpy as np
import sympy as sp
x = sp.symbols("x", positive=True)
print("(a)", sp.integrate(x**-3, (x, 1, sp.oo)))
print("(b)", sp.integrate(3 * sp.exp(-3 * x), (x, 0, sp.oo)))
print("(c)", sp.integrate(1 / sp.sqrt(x), (x, 1, sp.oo)))
print("(d)", sp.integrate(x * sp.exp(-x), (x, 0, sp.oo)))
b = sp.symbols("b", positive=True)
print("(c) integral from 1 to b:", sp.simplify(sp.integrate(1 / sp.sqrt(x), (x, 1, b))), "| at b = 1e6:", float((2 * sp.sqrt(b) - 2).subs(b, 10**6)))
```

```text
(a) 1/2
(b) 1
(c) oo
(d) 1
(c) integral from 1 to b: 2*sqrt(b) - 2 | at b = 1e6: 1998.0
```

## 9. A density

(a) The total area is $c\int_0^1 (1 - x^2)\,dx = c\left(1 - \frac13\right) = \frac{2c}{3}$, so $c = \frac32$. Then $E[X] = \frac32\int_0^1 (x - x^3)\,dx = \frac32\left(\frac12 - \frac14\right) = \frac38$, and $P(X \le \tfrac12) = \frac32\left[x - \frac{x^3}{3}\right]_0^{1/2} = \frac32\cdot\frac{11}{24} = \frac{11}{16}$. Since $f$ is larger near $0$, a mean below $\frac12$ and a probability above $\frac12$ for the left half make sense.

(b) $\int_0^b \lambda e^{-\lambda x}\,dx = 1 - e^{-\lambda b} \to 1$, so the total area is $1$. By parts and a limit, $E[X] = \int_0^\infty x\lambda e^{-\lambda x}\,dx = \frac1\lambda$. The probability is the area up to $a$: $P(X \le a) = 1 - e^{-\lambda a}$, which is the CDF (and its derivative $\lambda e^{-\lambda a}$ is the density, as Part 1 of the fundamental theorem says).

```python
import numpy as np
import sympy as sp
from scipy.integrate import quad
x, lam, c = sp.symbols("x lambda c", positive=True)
c_val = sp.solve(sp.Eq(sp.integrate(c * (1 - x**2), (x, 0, 1)), 1), c)[0]
f = c_val * (1 - x**2)
print("(a) c =", c_val, "| E[X] =", sp.integrate(x * f, (x, 0, 1)), "| P(X <= 1/2) =", sp.integrate(f, (x, 0, sp.Rational(1, 2))))
print("(b) total:", sp.integrate(lam * sp.exp(-lam * x), (x, 0, sp.oo)), "| E[X]:", sp.integrate(x * lam * sp.exp(-lam * x), (x, 0, sp.oo)))
a = sp.symbols("a", positive=True)
print("(b) P(X <= a):", sp.simplify(sp.integrate(lam * sp.exp(-lam * x), (x, 0, a))))
lam0 = 2.0
print("numeric check, lambda = 2: total", quad(lambda s: lam0 * np.exp(-lam0 * s), 0, np.inf)[0], "mean", quad(lambda s: s * lam0 * np.exp(-lam0 * s), 0, np.inf)[0])
```

```text
(a) c = 3/2 | E[X] = 3/8 | P(X <= 1/2) = 11/16
(b) total: 1 | E[X]: 1/lambda
(b) P(X <= a): 1 - exp(-a*lambda)
numeric check, lambda = 2: total 0.9999999999999999 mean 0.4999999999999997
```

## 10. Numerical rules in code

(a) and (b) The trapezoid error falls by a factor close to $4$ each time $n$ doubles, since it is proportional to $h^2$. Simpson's error falls by a factor close to $16$, since it is proportional to $h^4$. The first Simpson ratio is a little above $16$ because $n$ is still small, so the check uses a tolerance of $10\%$.

(c) `quad` returns the value together with an estimate of its own error, far smaller than anything the fixed-$n$ rules reach here. Always compare floats with `np.isclose`, never with `==`.

```python
import numpy as np
from scipy.integrate import quad

def trapezoid_rule(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    return (b - a) / n * (f(x).sum() - (f(a) + f(b)) / 2)

def simpson_rule(f, a, b, n):
    if n % 2:
        raise ValueError("Simpson needs an even number of strips")
    x = np.linspace(a, b, n + 1)
    return (b - a) / (3 * n) * (f(x[0]) + 4 * f(x[1:-1:2]).sum() + 2 * f(x[2:-1:2]).sum() + f(x[-1]))

ns = [4, 8, 16, 32]
trap_err = np.array([abs(trapezoid_rule(np.sin, 0, np.pi, n) - 2) for n in ns])
simp_err = np.array([abs(simpson_rule(np.sin, 0, np.pi, n) - 2) for n in ns])
for n, e1, e2 in zip(ns, trap_err, simp_err):
    print(f"n = {n:>2}   trapezoid error {e1:.3e}   simpson error {e2:.3e}")
print("trapezoid ratios:", trap_err[:-1] / trap_err[1:], "| near 4:", np.allclose(trap_err[:-1] / trap_err[1:], 4, rtol=0.05))
print("simpson ratios  :", simp_err[:-1] / simp_err[1:], "| near 16:", np.allclose(simp_err[:-1] / simp_err[1:], 16, rtol=0.1))
value, est = quad(np.sin, 0, np.pi)
print("quad:", value, "error estimate:", est, "| close to 2:", np.isclose(value, 2))
try:
    simpson_rule(np.sin, 0, np.pi, 5)
except ValueError as err:
    print("odd n raises:", err)
```

```text
n =  4   trapezoid error 1.039e-01   simpson error 4.560e-03
n =  8   trapezoid error 2.577e-02   simpson error 2.692e-04
n = 16   trapezoid error 6.430e-03   simpson error 1.659e-05
n = 32   trapezoid error 1.607e-03   simpson error 1.033e-06
trapezoid ratios: [4.03133722 4.00774118 4.00192956] | near 4: True
simpson ratios  : [16.94005966 16.22380632 16.05529226] | near 16: True
quad: 2.0 error estimate: 2.220446049250313e-14 | close to 2: True
odd n raises: Simpson needs an even number of strips
```
