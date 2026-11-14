# Solutions: Multivariable Calculus

Worked answers to the [problems](problems.md). The code outputs were produced by running the code. Numerical checks hold only up to rounding error, so the code compares with `np.isclose` and `np.allclose`, never `==`. All blocks share one session, so names such as `x`, `y`, and `t` carry over from earlier blocks. Back to the [topic](../README.md).

## 1. Partials by hand

(a) Treat $y$ as a constant for $f_x$ and $x$ as a constant for $f_y$: $f_x = 6xy + 5$ and $f_y = 3x^2 - 3y^2$. At $(1, 2)$ these are $17$ and $-9$. (b) $g_x = e^{2y}$ and $g_y = 2x\,e^{2y}$ (the chain rule on $e^{2y}$). (c) $h_x = yz$, $h_y = xz$, $h_z = xy + 2z$. (d) $f(1, 2+h) - f(1, 2) = (2+h)^2 - 4 = 4h + h^2$, so the quotient is $4 + h \to 4$.

```python
import numpy as np
import sympy as sp

x, y, z = sp.symbols("x y z")
fa = 3 * x**2 * y - y**3 + 5 * x
print("(a) f_x =", sp.diff(fa, x), "  f_y =", sp.diff(fa, y), "  at (1, 2):", sp.diff(fa, x).subs({x: 1, y: 2}), sp.diff(fa, y).subs({x: 1, y: 2}))
gb = x * sp.exp(2 * y)
print("(b) g_x =", sp.diff(gb, x), "  g_y =", sp.diff(gb, y))
hc = x * y * z + z**2
print("(c) h_x, h_y, h_z =", sp.diff(hc, x), ",", sp.diff(hc, y), ",", sp.diff(hc, z))
hh = sp.symbols("h")
print("(d) limit of ((1)(2+h)^2 - 4)/h =", sp.limit(((2 + hh) ** 2 - 4) / hh, hh, 0))
```

```text
(a) f_x = 6*x*y + 5   f_y = 3*x**2 - 3*y**2   at (1, 2): 17 -9
(b) g_x = exp(2*y)   g_y = 2*x*exp(2*y)
(c) h_x, h_y, h_z = y*z , x*z , x*y + 2*z
(d) limit of ((1)(2+h)^2 - 4)/h = 4
```

## 2. Mixed partials

$g_x = e^{y^2} - y\sin(xy)$, so $g_{xy} = 2y\,e^{y^2} - \sin(xy) - xy\cos(xy)$. And $g_y = 2xy\,e^{y^2} - x\sin(xy)$, so $g_{yx} = 2y\,e^{y^2} - \sin(xy) - xy\cos(xy)$. The same expression, as Clairaut's theorem promises for smooth functions.

```python
g2 = x * sp.exp(y**2) + sp.cos(x * y)
gxy, gyx = sp.diff(g2, x, y), sp.diff(g2, y, x)
print("g_xy =", gxy)
print("g_yx =", gyx)
print("difference simplifies to:", sp.simplify(gxy - gyx))
g2n = sp.lambdify((x, y), g2)
h = 1e-4
fx = lambda a, b: (g2n(a + h, b) - g2n(a - h, b)) / (2 * h)
fy = lambda a, b: (g2n(a, b + h) - g2n(a, b - h)) / (2 * h)
a0, b0 = 0.5, 1.2
n_xy = (fx(a0, b0 + h) - fx(a0, b0 - h)) / (2 * h)
n_yx = (fy(a0 + h, b0) - fy(a0 - h, b0)) / (2 * h)
print("numeric at (0.5, 1.2):", n_xy, n_yx, " equal:", np.isclose(n_xy, n_yx, atol=1e-5), " matches formula:", np.isclose(n_xy, float(gxy.subs({x: a0, y: b0})), atol=1e-5))
```

```text
g_xy = -x*y*cos(x*y) + 2*y*exp(y**2) - sin(x*y)
g_yx = -x*y*cos(x*y) + 2*y*exp(y**2) - sin(x*y)
difference simplifies to: 0
numeric at (0.5, 1.2): 9.069826312213536 9.069826312213536  equal: True  matches formula: True
```

## 3. Along a curve, and through two inner functions

(a) $f_x = 2x + 3y$, $f_y = 3x$, $x' = 2t$, $y' = 1$. At $t = 1$: $x = 1$, $y = 2$, so $f_x = 8$, $f_y = 3$, and $\frac{df}{dt} = 8\cdot 2 + 3\cdot 1 = 19$. Substituting, $f = t^4 + 3t^3 + 3t^2$ has derivative $4t^3 + 9t^2 + 6t = 19$ at $t = 1$.

(b) $f_u = 2uv$, $f_v = u^2$, and at $(1, 2)$ we have $u = 3$, $v = -1$. Then $\partial f/\partial x = f_u\cdot 1 + f_v\cdot 1 = -6 + 9 = 3$ and $\partial f/\partial y = f_u\cdot 1 + f_v\cdot(-1) = -6 - 9 = -15$.

```python
t = sp.symbols("t")
fc = x**2 + 3 * x * y
xt, yt = t**2, t + 1
chain = (sp.diff(fc, x) * sp.diff(xt, t) + sp.diff(fc, y) * sp.diff(yt, t)).subs({x: xt, y: yt})
print("(a) chain rule at t = 1:", chain.subs(t, 1), "   by substitution:", sp.diff(fc.subs({x: xt, y: yt}), t).subs(t, 1))
u_, v_ = sp.symbols("u v")
f_uv = u_**2 * v_
U, V = x + y, x - y
dfx = (sp.diff(f_uv, u_) * sp.diff(U, x) + sp.diff(f_uv, v_) * sp.diff(V, x)).subs({u_: U, v_: V})
dfy = (sp.diff(f_uv, u_) * sp.diff(U, y) + sp.diff(f_uv, v_) * sp.diff(V, y)).subs({u_: U, v_: V})
direct = f_uv.subs({u_: U, v_: V})
print("(b) df/dx at (1, 2):", dfx.subs({x: 1, y: 2}), " direct:", sp.diff(direct, x).subs({x: 1, y: 2}))
print("    df/dy at (1, 2):", dfy.subs({x: 1, y: 2}), " direct:", sp.diff(direct, y).subs({x: 1, y: 2}))
```

```text
(a) chain rule at t = 1: 19    by substitution: 19
(b) df/dx at (1, 2): 3  direct: 3
    df/dy at (1, 2): -15  direct: -15
```

## 4. Gradient and directional derivative

$\nabla f = (2xy,\; x^2 + 2y) = (4, 5)$ at $(1, 2)$. The unit vector along $(3, 4)$ is $(0.6, 0.8)$, so the slope is $4(0.6) + 5(0.8) = 6.4$. The fastest increase is along $\nabla f/\lVert\nabla f\rVert = (4, 5)/\sqrt{41}$, at the rate $\lVert\nabla f\rVert = \sqrt{41}\approx 6.40$. A direction with no change is perpendicular to the gradient, for example $(5, -4)/\sqrt{41}$.

```python
f4 = x**2 * y + y**2
grad4 = sp.Matrix([sp.diff(f4, v) for v in (x, y)]).subs({x: 1, y: 2})
g4 = np.array(grad4, dtype=float).ravel()
u4 = np.array([3.0, 4.0]) / 5
print("(a) gradient at (1, 2):", g4)
print("(b) D_u f for u = (0.6, 0.8):", g4 @ u4)
print("(c) steepest ascent direction:", np.round(g4 / np.linalg.norm(g4), 4), "  rate:", np.linalg.norm(g4), " sqrt(41) =", np.sqrt(41))
zero = np.array([g4[1], -g4[0]]) / np.linalg.norm(g4)
print("(d) a zero-change direction:", np.round(zero, 4), "  slope there:", np.round(g4 @ zero, 12))
```

```text
(a) gradient at (1, 2): [4. 5.]
(b) D_u f for u = (0.6, 0.8): 6.4
(c) steepest ascent direction: [0.6247 0.7809]   rate: 6.4031242374328485  sqrt(41) = 6.4031242374328485
(d) a zero-change direction: [ 0.7809 -0.6247]   slope there: 0.0
```

## 5. A Jacobian

(a) $`J_F = \begin{pmatrix} e^x\cos y & -e^x\sin y \\ e^x\sin y & e^x\cos y\end{pmatrix}`$ and $\det J_F = e^{2x}(\cos^2 y + \sin^2 y) = e^{2x}$. At $(0, \pi/2)$ the matrix is a quarter-turn rotation, with determinant $1$. (b) $J_G = (2u,\; 2v)$, evaluated at $F(x, y)$. The product $J_G\,J_F$ simplifies to $(2e^{2x},\; 0)$. Directly, $G(F(x, y)) = e^{2x}$, whose gradient is $(2e^{2x}, 0)$.

```python
F5 = sp.Matrix([sp.exp(x) * sp.cos(y), sp.exp(x) * sp.sin(y)])
J5 = F5.jacobian([x, y])
print("(a) Jacobian:", sp.simplify(J5), "  det:", sp.simplify(J5.det()))
print("    at (0, pi/2):", J5.subs({x: 0, y: sp.pi / 2}), "  det:", J5.subs({x: 0, y: sp.pi / 2}).det())
Jg = sp.Matrix([[2 * u_, 2 * v_]])
JgF = Jg.subs({u_: F5[0], v_: F5[1]})
prod = sp.simplify(JgF * J5)
print("(b) J_G(F) * J_F =", prod, "   gradient of exp(2x):", sp.Matrix([sp.diff(sp.exp(2 * x), x), 0]).T)
```

```text
(a) Jacobian: Matrix([[exp(x)*cos(y), -exp(x)*sin(y)], [exp(x)*sin(y), exp(x)*cos(y)]])   det: exp(2*x)
    at (0, pi/2): Matrix([[0, -1], [1, 0]])   det: 1
(b) J_G(F) * J_F = Matrix([[2*exp(2*x), 0]])    gradient of exp(2x): Matrix([[2*exp(2*x), 0]])
```

## 6. Classify the critical points

$\nabla f = (3x^2 - 3y,\; 3y^2 - 3x) = \mathbf{0}$. From the first equation $y = x^2$, and substituting gives $x^4 = x$, so $x = 0$ or $x = 1$ (the other roots of $x^3 = 1$ are complex). The critical points are $(0, 0)$ and $(1, 1)$. The Hessian is $`\begin{pmatrix} 6x & -3 \\ -3 & 6y\end{pmatrix}`$. At $(0, 0)$ it has eigenvalues $\pm3$: a saddle. At $(1, 1)$ the eigenvalues are $3$ and $9$: a minimum, with condition number $9/3 = 3$.

```python
f6 = x**3 + y**3 - 3 * x * y
H6 = sp.hessian(f6, (x, y))
for c in sp.solve([sp.diff(f6, v) for v in (x, y)], [x, y], dict=True):
    if not all(val.is_real for val in c.values()):
        continue
    Hn = np.array(H6.subs(c), dtype=float)
    eig = np.linalg.eigvalsh(Hn)
    kind = "minimum" if np.all(eig > 0) else "maximum" if np.all(eig < 0) else "saddle"
    print(c, " H =", Hn.tolist(), " eigenvalues:", eig, "->", kind, "  cond:", round(eig.max() / eig.min(), 3) if kind == "minimum" else "")
```

```text
{x: 0, y: 0}  H = [[0.0, -3.0], [-3.0, 0.0]]  eigenvalues: [-3.  3.] -> saddle   cond: 
{x: 1, y: 1}  H = [[6.0, -3.0], [-3.0, 6.0]]  eigenvalues: [3. 9.] -> minimum   cond: 3.0
```

## 7. Linear and quadratic approximations

At $(3, 4)$: $f = 5$ and $\nabla f = (3/5, 4/5)$. The step is $\boldsymbol\Delta = (0.04, -0.03)$, and $\nabla f\cdot\boldsymbol\Delta = 0.024 - 0.024 = 0$, so the linear estimate stays at $5$. The Hessian term adds $\frac12\boldsymbol\Delta^\top H\boldsymbol\Delta$, which for a step perpendicular to the gradient equals $\frac{\lVert\boldsymbol\Delta\rVert^2}{2r} = \frac{0.0025}{10} = 0.00025$. The quadratic estimate is $5.00025$, and the true value is about $5.00024999$.

```python
r = sp.sqrt(x**2 + y**2)
a7 = {x: 3, y: 4}
grad7 = np.array([float(sp.diff(r, v).subs(a7)) for v in (x, y)])
H7 = np.array(sp.hessian(r, (x, y)).subs(a7), dtype=float)
step = np.array([0.04, -0.03])
true = float(r.subs({x: 3.04, y: 3.97}))
lin = 5 + grad7 @ step
quad = lin + 0.5 * step @ H7 @ step
print("gradient:", grad7, " Hessian:", np.round(H7, 4).tolist())
print("true:", true, " linear:", lin, " quadratic:", quad)
print("errors:", abs(true - lin), abs(true - quad))
```

```text
gradient: [0.6 0.8]  Hessian: [[0.128, -0.096], [-0.096, 0.072]]
true: 5.000249993750313  linear: 5.0  quadratic: 5.00025
errors: 0.0002499937503133154 6.249686990145165e-09
```

## 8. Iterated integrals

(a) $\int_0^2 (3x^2 + 2xy)\,dy = 6x^2 + 4x$, then $\int_0^1 (6x^2 + 4x)\,dx = 2 + 2 = 4$. (b) For each $x$, $y$ runs from $x^2$ up to $x$: $\int_0^1\int_{x^2}^{x} xy\,dy\,dx = \int_0^1 \frac{x(x^2 - x^4)}{2}\,dx = \frac12\left(\frac14 - \frac16\right) = \frac1{24}$. Swapped, each $y$ in $[0, 1]$ has $x$ running from $y$ to $\sqrt{y}$: $\int_0^1\int_y^{\sqrt y} xy\,dx\,dy = \int_0^1\frac{y(y - y^2)}{2}\,dy = \frac12\left(\frac13 - \frac14\right) = \frac1{24}$.

```python
print("(a)", sp.integrate(3 * x**2 + 2 * x * y, (y, 0, 2), (x, 0, 1)))
print("(b) dy first:", sp.integrate(x * y, (y, x**2, x), (x, 0, 1)), "  dx first:", sp.integrate(x * y, (x, y, sp.sqrt(y)), (y, 0, 1)))
```

```text
(a) 4
(b) dy first: 1/24   dx first: 1/24
```

## 9. Polar coordinates

(a) $\int_0^{2\pi}\int_0^2 e^{-r^2}\,r\,dr\,d\theta = 2\pi\cdot\frac{1 - e^{-4}}{2} = \pi(1 - e^{-4})$. (b) $\iint (x^2 + y^2)\,dA = \int_0^{2\pi}\int_0^1 r^3\,dr\,d\theta = \frac{\pi}{2}$, and dividing by the area $\pi$ gives an average of $\frac12$. (c) With $xy = r^2\cos\theta\sin\theta$ and $\theta$ from $0$ to $\pi/2$: $\int_0^{\pi/2}\cos\theta\sin\theta\,d\theta\cdot\int_0^1 r^3\,dr = \frac12\cdot\frac14 = \frac18$.

```python
rr, th = sp.symbols("r theta", positive=True)
print("(a)", sp.integrate(sp.exp(-rr**2) * rr, (rr, 0, 2), (th, 0, 2 * sp.pi)), "=", float(sp.pi * (1 - sp.exp(-4))))
total = sp.integrate(rr**2 * rr, (rr, 0, 1), (th, 0, 2 * sp.pi))
print("(b) integral of x^2 + y^2 over the unit disc:", total, "  average:", sp.simplify(total / sp.pi))
print("(c)", sp.integrate(rr**2 * sp.cos(th) * sp.sin(th) * rr, (rr, 0, 1), (th, 0, sp.pi / 2)))
```

```text
(a) 2*pi*(1/2 - exp(-4)/2) = 3.0840523770111425
(b) integral of x^2 + y^2 over the unit disc: pi/2   average: 1/2
(c) 1/8
```

## 10. Numerical gradient and Hessian

The Hessian is the Jacobian of the gradient, so each column is a centered difference of `num_grad`. The two critical points are classified by the eigenvalues, as in problem 6.

```python
def num_grad(func, p, h=1e-5):
    p = np.asarray(p, dtype=float)
    g = np.zeros_like(p)
    for i in range(len(p)):
        e = np.zeros_like(p)
        e[i] = h
        g[i] = (func(p + e) - func(p - e)) / (2 * h)
    return g

def num_hessian(func, p, h=1e-4):
    p = np.asarray(p, dtype=float)
    n = len(p)
    H = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        H[:, j] = (num_grad(func, p + e) - num_grad(func, p - e)) / (2 * h)   # differentiate the gradient
    return 0.5 * (H + H.T)                                                  # remove tiny asymmetry from rounding

def classify(H):
    eig = np.linalg.eigvalsh(H)
    return "minimum" if np.all(eig > 0) else "maximum" if np.all(eig < 0) else "saddle" if eig.min() < 0 < eig.max() else "inconclusive"

f10 = lambda p: p[0]**3 + p[1]**3 - 3 * p[0] * p[1]
for pt in [(0.0, 0.0), (1.0, 1.0)]:
    H_num = num_hessian(f10, pt)
    H_exact = np.array(sp.hessian(x**3 + y**3 - 3 * x * y, (x, y)).subs({x: pt[0], y: pt[1]}), dtype=float)
    print(pt, "gradient ~ 0:", np.allclose(num_grad(f10, pt), 0, atol=1e-6), " Hessian matches sympy:", np.allclose(H_num, H_exact, atol=1e-4), "->", classify(H_num))

f3 = lambda p: p[0]**2 + 2 * p[1]**2 + 3 * p[2]**2 + p[0] * p[1]
print("3-variable gradient at (1, 1, 1):", np.round(num_grad(f3, [1, 1, 1]), 5), " exact (3, 5, 6):", np.allclose(num_grad(f3, [1, 1, 1]), [3, 5, 6]))
```

```text
(0.0, 0.0) gradient ~ 0: True  Hessian matches sympy: True -> saddle
(1.0, 1.0) gradient ~ 0: True  Hessian matches sympy: True -> minimum
3-variable gradient at (1, 1, 1): [3. 5. 6.]  exact (3, 5, 6): True
```
