# Solutions: Gradient Descent and its Variants

Worked answers to the [problems](problems.md). The code outputs were produced by running the code. Numerical checks hold only up to rounding error, so the code compares with `np.isclose` and `np.allclose`, never `==`. All blocks share one session, so names such as `np` and `gradient_descent` carry over from earlier blocks. Back to the [topic](../README.md).

## 1. Three steps by hand

(a) $f'(x) = 2x - 6$, so $x_1 = 0 - 0.2(-6) = 1.2$, $x_2 = 1.2 - 0.2(-3.6) = 1.92$, and $x_3 = 1.92 - 0.2(-2.16) = 2.352$. (b) The minimum is at $x = 3$. The update is $x - 3 \leftarrow (1 - 2\eta)(x - 3) = 0.6\,(x - 3)$, so the distance is multiplied by $0.6$ each step: $3, 1.8, 1.08, 0.648$. (c) We need $3\cdot0.6^k < 0.01$, so $k > \ln(300)/\ln(1/0.6) \approx 11.17$, which means $12$ steps.

```python
import numpy as np

def gradient_descent(grad, x0, eta, steps):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(steps):
        x = x - eta * grad(x)
        path.append(x.copy())
    return np.array(path)

path = gradient_descent(lambda x: 2 * x - 6, [0.0], 0.2, 12)[:, 0]
print("x_0..x_3:", path[:4])
print("distance factor is 0.6:", np.allclose(np.abs(path[1:] - 3) / np.abs(path[:-1] - 3), 0.6))
print("smallest k with distance < 0.01:", int(np.argmax(np.abs(path - 3) < 0.01)))
```

```text
x_0..x_3: [0.    1.2   1.92  2.352]
distance factor is 0.6: True
smallest k with distance < 0.01: 12
```

## 2. The learning-rate limit

(a) $\operatorname{tr} H = 8$ and $\det H = 11$, so $\lambda = 4 \pm \sqrt{5}$, about $1.764$ and $6.236$. (b) We need $\eta < 2/\lambda_{\max} = 2/6.236 \approx 0.3207$. (c) The best constant step is $\eta = 2/(\lambda_{\max} + \lambda_{\min}) = 2/8 = 0.25$. Here $\kappa = 6.236/1.764 \approx 3.535$, and the factor is $(\kappa - 1)/(\kappa + 1) = (\lambda_{\max} - \lambda_{\min})/(\lambda_{\max} + \lambda_{\min}) = 2\sqrt5/8 \approx 0.559$.

```python
H = np.array([[6.0, 1.0], [1.0, 2.0]])
lam = np.linalg.eigvalsh(H)
print("eigenvalues:", lam, "  kappa:", lam[-1] / lam[0])
print("limit 2/lambda_max:", 2 / lam[-1], "  best eta:", 2 / lam.sum(), "  factor:", (lam[-1] - lam[0]) / lam.sum())
grad2 = lambda p: H @ p                                   # f = (1/2) p^T H p has gradient H p
for eta in [0.32, 0.33]:
    print(eta, "-> distance after 200 steps:", np.linalg.norm(gradient_descent(grad2, [1, 1], eta, 200)[-1]))
```

```text
eigenvalues: [1.76393202 6.23606798]   kappa: 3.5353221654543927
limit 2/lambda_max: 0.3207149131818564   best eta: 0.25   factor: 0.5590169943749475
0.32 -> distance after 200 steps: 0.4922232331830219
0.33 -> distance after 200 steps: 93194.32551526505
```

(The loss in the problem is $\frac12\mathbf{p}^\top H\mathbf{p}$ with this $H$, which is $3x^2 + xy + y^2$.) Just below the limit the iterate shrinks (slowly), and just above it blows up.

## 3. A condition-number question

(a) $\nabla f = (2x, 200y)$, so $H = \operatorname{diag}(2, 200)$, $\kappa = 100$, and $\eta < 2/200 = 0.01$. (b) The best step is $\eta = 2/202 \approx 0.0099$, with factor $99/101 \approx 0.980$. We need $0.980^k < 10^{-3}$, so $k > \ln(10^{-3})/\ln(0.980) \approx 345$ steps. (c) With $y = u/10$ the loss is $x^2 + u^2$, so $H = \operatorname{diag}(2, 2)$ and $\kappa = 1$. Rescaling the second coordinate turned the valley into a bowl. That is what standardising the features does for a regression loss: gradient descent with $\eta = 1/2$ would then solve it in one step.

```python
print("best eta:", 2 / 202, "  factor:", 99 / 101, "  steps for 1000-fold:", np.log(1e-3) / np.log(99 / 101))
```

```text
best eta: 0.009900990099009901   factor: 0.9801980198019802   steps for 1000-fold: 345.3762507166141
```

## 4. Cost and noise

(a) A full-batch update costs $n\cdot 2d = 2\,000\,000\cdot 1000 = 2\times10^9$ multiply-adds. For $B = 64$ it is $64\,000$, and for $B = 1$ it is $1000$. (b) One epoch is $n/B$ updates: $1$ for the full batch, $31\,250$ for $B = 64$, and $2\,000\,000$ for $B = 1$. Every epoch costs the same total, about $2\times10^9$ multiply-adds, but the mini-batch methods make many more updates for it. (c) The standard deviation falls like $1/\sqrt{B}$: $40/\sqrt{64} = 5$. To reach $1$ we need $\sqrt{B} = 40$, so $B = 1600$.

```python
n, d, sigma = 2_000_000, 500, 40.0
for B in [n, 64, 1]:
    print(f"B = {B:>9,}: cost per update {B * 2 * d:>13,}, updates per epoch {n // B:>9,}")
print("std for B = 64:", sigma / np.sqrt(64), "  B needed for std 1:", (sigma / 1.0) ** 2)
```

```text
B = 2,000,000: cost per update 2,000,000,000, updates per epoch         1
B =        64: cost per update        64,000, updates per epoch    31,250
B =         1: cost per update         1,000, updates per epoch 2,000,000
std for B = 64: 5.0   B needed for std 1: 1600.0
```

## 5. A momentum step by hand

The gradient is $(x, 10y)$. Step 1: $\mathbf{v} = (2, 10)$, so $\mathbf{x} = (2, 1) - 0.05(2, 10) = (1.9, 0.5)$. Step 2: the gradient is $(1.9, 5)$, so $\mathbf{v} = 0.8(2, 10) + (1.9, 5) = (3.5, 13)$ and $\mathbf{x} = (1.9, 0.5) - 0.05(3.5, 13) = (1.725, -0.15)$. Step 3: the gradient is $(1.725, -1.5)$, so $\mathbf{v} = 0.8(3.5, 13) + (1.725, -1.5) = (4.525, 8.9)$ and $\mathbf{x} = (1.4988, -0.595)$. Plain descent gives $(1.9, 0.5)$ and then $(1.805, 0.25)$. Momentum already moves $x$ further (the gentle direction, where gradients keep agreeing). The steep coordinate $y$ overshoots through zero to $-0.15$ and then $-0.595$, because its velocity is still pointing down when the gradient has changed sign. That swing, which damps down over a few more steps, is the price of the extra speed.

```python
g5 = lambda p: np.array([1.0, 10.0]) * p
p, v = np.array([2.0, 1.0]), np.zeros(2)
for k in range(3):
    v = 0.8 * v + g5(p)
    p = p - 0.05 * v
    print("momentum:", v, p)
print("plain GD:", gradient_descent(g5, [2, 1], 0.05, 2)[1:])
```

```text
momentum: [ 2. 10.] [1.9 0.5]
momentum: [ 3.5 13. ] [ 1.725 -0.15 ]
momentum: [4.525 8.9  ] [ 1.49875 -0.595  ]
plain GD: [[1.9   0.5  ]
 [1.805 0.25 ]]
```

## 6. Adam's first steps by hand

Step 1, $\mathbf{g} = (4, -0.5)$: $\mathbf{m}_1 = 0.1\mathbf{g} = (0.4, -0.05)$ and $\mathbf{v}_1 = 0.001\mathbf{g}^2 = (0.016, 0.00025)$. The corrections divide by $1 - 0.9 = 0.1$ and $1 - 0.999 = 0.001$, giving $\hat{\mathbf{m}}_1 = (4, -0.5)$ and $\hat{\mathbf{v}}_1 = (16, 0.25)$. So the step is $\eta\,\hat{\mathbf{m}}_1/\sqrt{\hat{\mathbf{v}}_1} = 0.01\,(4/4, -0.5/0.5) = (0.01, -0.01)$, and the position is $(0.99, 1.01)$. Both coordinates move by exactly $\eta$, because $\hat{\mathbf{m}}_1 = \mathbf{g}$ and $\sqrt{\hat{\mathbf{v}}_1} = \lvert\mathbf{g}\rvert$: the first step is $\eta\cdot\operatorname{sign}(\mathbf{g})$ whatever the gradient's size.

Step 2, $\mathbf{g} = (2, -0.5)$: $\mathbf{m}_2 = 0.9(0.4, -0.05) + 0.1(2, -0.5) = (0.56, -0.095)$, and $\mathbf{v}_2 = 0.999(0.016, 0.00025) + 0.001(4, 0.25) \approx (0.019984, 0.00049975)$. With divisors $1 - 0.9^2 = 0.19$ and $1 - 0.999^2 = 0.001999$: $\hat{\mathbf{m}}_2 \approx (2.947, -0.5)$ and $\hat{\mathbf{v}}_2 \approx (9.997, 0.25)$. The step is about $0.01\,(0.932, -1)$, so the position is about $(0.9807, 1.02)$. The first coordinate's gradient fell, and its step shrank slightly.

```python
m, v, w = np.zeros(2), np.zeros(2), np.array([1.0, 1.0])
for t, g in enumerate([np.array([4.0, -0.5]), np.array([2.0, -0.5])], start=1):
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g**2
    m_hat, v_hat = m / (1 - 0.9**t), v / (1 - 0.999**t)
    step = 0.01 * m_hat / np.sqrt(v_hat)
    w = w - step
    print(f"t={t}: m={m}, v={v}\n      m_hat={m_hat}, v_hat={v_hat}\n      step={step}, w={w}")
```

```text
t=1: m=[ 0.4  -0.05], v=[0.016   0.00025]
      m_hat=[ 4.  -0.5], v_hat=[16.    0.25]
      step=[ 0.01 -0.01], w=[0.99 1.01]
t=2: m=[ 0.56  -0.095], v=[0.019984   0.00049975]
      m_hat=[ 2.94736842 -0.5       ], v_hat=[9.9969985 0.25     ]
      step=[ 0.0093218 -0.01     ], w=[0.9806782 1.02     ]
```

## 7. A schedule

(a) During warm-up $\eta_t = \eta_0(t+1)/W$, so $\eta_{49} = 0.003\cdot 50/100 = 0.0015$. At $t = 100$ the cosine starts at its top: $\eta_{100} = 0.003$. Halfway through the decay, $t = 550$, we are $450$ of $900$ steps in, and $\cos(\pi/2) = 0$, so $\eta_{550} = 0.0015$. At the end $\eta_{1000} = 0$. (b) $n/B = 500$ updates per epoch, so $1000$ updates are $2$ epochs. (c) Early gradients are erratic and the second-moment averages are still short of data, so small first steps keep the first updates from throwing the weights far off. The decay at the end shrinks the step so that the noise of mini-batches stops kicking the weights around and the run settles.

```python
eta0, W, T = 0.003, 100, 1000
sched = lambda t: eta0 * (t + 1) / W if t < W else 0.5 * eta0 * (1 + np.cos(np.pi * (t - W) / (T - W)))
for t in [49, 100, 550, 1000]:
    print(f"eta at update {t}: {sched(t):.6f}")
print("updates per epoch:", 50_000 // 100, "  epochs in 1000 updates:", 1000 / (50_000 // 100))
```

```text
eta at update 49: 0.001500
eta at update 100: 0.003000
eta at update 550: 0.001500
eta at update 1000: 0.000000
updates per epoch: 500   epochs in 1000 updates: 2.0
```

## 8. Which optimiser?

(a) Do not use gradient descent at all: with $20$ features the normal equations (or `np.linalg.lstsq`) give the exact answer in one cheap solve. If you do want descent, standardise the features and use full-batch gradient descent. (b) AdaGrad, or Adam, on mini-batches: sparse features are active rarely, and per-coordinate step sizes let the rare features take large steps. Stochastic updates also avoid reading all $10^6$ columns for every step. (c) AdamW with a short linear warm-up and then a cosine (or linear) decay, which is the common recipe for Transformers. (d) The loss exploded, so the learning rate is too large (or the gradient has a bug). Lower $\eta$ by a factor of $3$ to $10$, add warm-up, and run a gradient check on a tiny batch.

## 9. A plateau

(a) $f'(x) = 4x^3$, so $x_1 = 1 - 0.05\cdot 4 = 0.8$, $x_2 = 0.8 - 0.05\cdot 4\cdot0.512 = 0.6976$, and $x_3 = 0.6976 - 0.05\cdot4\cdot0.33948 \approx 0.6297$. (b) The step is $\eta\cdot4x^3$, which shrinks like $x^3$ while the remaining distance shrinks only like $x$. The curvature $f'' = 12x^2$ also goes to $0$, so the bottom is very flat: the gradient is tiny long before $x$ is. (c) The code below stops at step $624$ with $x \approx 0.063$, because $4x^3 < 10^{-3}$ there. The minimum is at $0$, so the loop stopped $0.063$ away from it, with $f$ still about $1.6\times10^{-5}$ above the minimum value. A gradient-norm rule measures the slope, not the distance, so on flat functions use a tighter tolerance or also watch the loss and the step count.

```python
x = 1.0
xs = []
for k in range(1, 5001):
    x = x - 0.05 * 4 * x**3
    xs.append(x)
    if abs(4 * x**3) < 1e-3:
        print("stopped at step", k, " x =", x, " f =", x**4)
        break
print("x_1, x_2, x_3:", np.round(xs[:3], 4))
```

```text
stopped at step 624  x = 0.06295347813361159  f = 1.5706481902392806e-05
x_1, x_2, x_3: [0.8    0.6976 0.6297]
```

## 10. Compare three optimisers

Each method is a small function that takes the position, the gradient, and its own state, and returns the new position and state.

```python
H10 = np.array([2.0, 200.0])
f10 = lambda p: 0.5 * np.sum(H10 * p**2)                  # x^2 + 100 y^2
grad10 = lambda p: H10 * p

def gd_update(p, g, s, eta):
    return p - eta * g, s

def momentum_update(p, g, s, eta, beta):
    v = beta * s + g
    return p - eta * v, v

def adam_update(p, g, s, eta, b1=0.9, b2=0.999, eps=1e-8):
    m, v, t = s
    t += 1
    m, v = b1 * m + (1 - b1) * g, b2 * v + (1 - b2) * g**2
    return p - eta * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps), (m, v, t)

def steps_until(update, state, f, grad, p0, tol=1e-8, cap=100_000, **kw):
    p = np.array(p0, dtype=float)
    for k in range(cap):
        if f(p) < tol:
            return k
        p, state = update(p, grad(p), state, **kw)
    return None

print("gradient descent:", steps_until(gd_update, None, f10, grad10, [5, 1], eta=2 / 202))
print("momentum        :", steps_until(momentum_update, np.zeros(2), f10, grad10, [5, 1], eta=0.0165, beta=0.67))
print("Adam            :", steps_until(adam_update, (np.zeros(2), np.zeros(2), 0), f10, grad10, [5, 1], eta=0.1))

def num_grad(func, p, h=1e-5):
    p = np.asarray(p, dtype=float); g = np.zeros_like(p)
    for i in range(len(p)):
        e = np.zeros_like(p); e[i] = h
        g[i] = (func(p + e) - func(p - e)) / (2 * h)
    return g

print("gradient check:", np.allclose(num_grad(f10, [1, 1]), grad10(np.array([1.0, 1.0]))))
fu = lambda p: 0.5 * np.sum(np.array([2.0, 2.0]) * p**2)  # x^2 + u^2 in the rescaled coordinates
print("rescaled, eta = 0.5:", steps_until(gd_update, None, fu, lambda p: 2 * p, [5, 10], eta=0.5), "step(s)")
```

```text
gradient descent: 582
momentum        : 67
Adam            : 224
gradient check: True
rescaled, eta = 0.5: 1 step(s)
```

The counts show the pattern of the README. Gradient descent needs several hundred steps because $\kappa = 100$. Momentum with the heavy-ball settings needs roughly a ninth of that. Adam lands in between without being told the curvature, and with a constant $\eta$ it oscillates around the minimum before settling. In the rescaled coordinates the condition number is $1$ and a single step with $\eta = 1/\lambda = 0.5$ lands on the minimum.
