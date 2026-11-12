# Solutions: Projections and Least Squares

Worked answers to the [problems](problems.md). The code outputs were produced by running the code. Numerical checks hold only up to rounding error, so the code compares with `np.allclose`. Back to the [topic](../README.md).

## 1. Project

(a) $\mathbf{a}\cdot\mathbf{b} = 6$ and $\mathbf{a}\cdot\mathbf{a} = 3$, so $\hat{x} = 2$ and $\mathbf{p} = (2,2,2)$. The error is $(1,2,3) - (2,2,2) = (-1, 0, 1)$, and $(1,1,1)\cdot(-1,0,1) = 0$.

(b) $\mathbf{a}\cdot\mathbf{b} = 2 - 2 + 6 = 6$ and $\mathbf{a}\cdot\mathbf{a} = 9$, so $\hat{x} = 2/3$ and $\mathbf{p} = \tfrac23(2,-1,2) = (\tfrac43, -\tfrac23, \tfrac43)$. The error is $(1,2,3) - \mathbf{p} = (-\tfrac13, \tfrac83, \tfrac53)$, and $\mathbf{a}\cdot\mathbf{e} = -\tfrac23 - \tfrac83 + \tfrac{10}3 = 0$.

```python
import numpy as np

b = np.array([1.0, 2.0, 3.0])
for a in (np.array([1.0, 1.0, 1.0]), np.array([2.0, -1.0, 2.0])):
    p = (a @ b) / (a @ a) * a
    print(p.round(4), (b - p).round(4), np.isclose(a @ (b - p), 0))
```

```text
[2. 2. 2.] [-1.  0.  1.] True
[ 1.3333 -0.6667  1.3333] [-0.3333  2.6667  1.6667] True
```

## 2. A projection matrix for a line

With $\mathbf{a} = (1,2)$, $\mathbf{a}^\top\mathbf{a} = 5$ and $\mathbf{a}\mathbf{a}^\top$ has rows $(1,2)$ and $(2,4)$, so

```math
P = \frac15\begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}.
```

It is symmetric. $P^2$ has first row $\tfrac1{25}(1 + 4,\ 2 + 8) = \tfrac15(1, 2)$ and second row $\tfrac1{25}(2 + 8,\ 4 + 16) = \tfrac15(2, 4)$, which is $P$. The trace is $(1 + 4)/5 = 1$, the dimension of the line. And $P\mathbf{b} = \tfrac15(4 + 4,\ 8 + 8) = (1.6, 3.2)$.

```python
a = np.array([1.0, 2.0])
P = np.outer(a, a) / (a @ a)
print(5 * P)
print(np.allclose(P @ P, P), np.allclose(P.T, P), np.trace(P))
print(P @ np.array([4.0, 2.0]))
```

```text
[[1. 2.]
 [2. 4.]]
True True 1.0
[1.6 3.2]
```

## 3. Normal equations

(a) The columns are $(1,0,1)$ and $(0,1,1)$, so
```math
A^\top A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}, \qquad A^\top\mathbf{b} = \begin{bmatrix} 1 + 4 \\ 2 + 4 \end{bmatrix} = \begin{bmatrix} 5 \\ 6 \end{bmatrix}.
```
The equations $2\hat{x}_1 + \hat{x}_2 = 5$ and $\hat{x}_1 + 2\hat{x}_2 = 6$ give (multiply the first by 2 and subtract the second) $3\hat{x}_1 = 4$, so $\hat{x}_1 = \tfrac43$ and $\hat{x}_2 = 5 - \tfrac83 = \tfrac73$.

(b) $\mathbf{p} = \tfrac43(1,0,1) + \tfrac73(0,1,1) = (\tfrac43, \tfrac73, \tfrac{11}3)$ and $\mathbf{e} = \mathbf{b} - \mathbf{p} = (-\tfrac13, -\tfrac13, \tfrac13)$. Check: $A^\top\mathbf{e}$ has entries $-\tfrac13 + \tfrac13 = 0$ and $-\tfrac13 + \tfrac13 = 0$.

(c) No. The error is not zero, so $\mathbf{b}$ differs from its projection and is not in the column space. (The column space is the plane $z = x + y$, and $4 \ne 1 + 2$.)

```python
A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
b = np.array([1.0, 2.0, 4.0])
x_hat = np.linalg.solve(A.T @ A, A.T @ b)
p = A @ x_hat
e = b - p
print(x_hat, p, e)
print(np.allclose(A.T @ e, 0))
```

```text
[1.33333333 2.33333333] [1.33333333 2.33333333 3.66666667] [-0.33333333 -0.33333333  0.33333333]
True
```

## 4. The projection matrix

$`(A^\top A)^{-1} = \tfrac13\begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}`$, and multiplying out $A(A^\top A)^{-1}A^\top$ gives

```math
P = \frac13\begin{bmatrix} 2 & -1 & 1 \\ -1 & 2 & 1 \\ 1 & 1 & 2 \end{bmatrix}.
```

The trace is $(2 + 2 + 2)/3 = 2$, the dimension of the column space. $P\mathbf{b} = \tfrac13(2 - 2 + 4,\ -1 + 4 + 4,\ 1 + 2 + 8) = (\tfrac43, \tfrac73, \tfrac{11}3) = \mathbf{p}$. The matrix $I - P$ has trace $3 - 2 = 1$, and it projects onto the left nullspace $N(A^\top)$, a line (spanned by $(-1,-1,1)$, the direction of $\mathbf{e}$).

```python
P = A @ np.linalg.inv(A.T @ A) @ A.T
print(3 * P)
print(np.allclose(P @ b, p), np.allclose(P @ P, P), np.trace(P))
print(np.trace(np.eye(3) - P), np.allclose((np.eye(3) - P) @ b, e))
```

```text
[[ 2. -1.  1.]
 [-1.  2.  1.]
 [ 1.  1.  2.]]
True True 2.0
1.0 True
```

## 5. Why $I - P$ is a projection too

$(I - P)^2 = I - 2P + P^2 = I - 2P + P = I - P$, using $P^2 = P$. And $(I - P)^\top = I - P^\top = I - P$, using $P^\top = P$. Also $P(I - P) = P - P^2 = P - P = 0$. In words: $(I - P)\mathbf{b}$ is the error $\mathbf{e}$, and $P\mathbf{e} = \mathbf{0}$ says that projecting the error onto the subspace gives nothing, since the error is already perpendicular to it.

## 6. A line fit by hand

The design matrix has rows $(1,1)$, $(1,2)$, $(1,3)$, $(1,4)$ and the right-hand side is $\mathbf{y} = (2,3,5,4)$. Then
```math
A^\top A = \begin{bmatrix} 4 & 10 \\ 10 & 30 \end{bmatrix}, \qquad A^\top\mathbf{y} = \begin{bmatrix} 14 \\ 39 \end{bmatrix}.
```
Solve $4c + 10d = 14$ and $10c + 30d = 39$: multiply the first by $2.5$ to get $10c + 25d = 35$, and subtract from the second to get $5d = 4$, so $d = 0.8$ and $c = (14 - 8)/4 = 1.5$. The line is $\hat{y} = 1.5 + 0.8t$. The fitted values are $(2.3, 3.1, 3.9, 4.7)$ and the residuals are $(-0.3, -0.1, 1.1, -0.7)$. They sum to $0$, and $\sum t_i r_i = -0.3 - 0.2 + 3.3 - 2.8 = 0$. The sum of squared residuals is $0.09 + 0.01 + 1.21 + 0.49 = 1.8$.

```python
t = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([2.0, 3.0, 5.0, 4.0])
A6 = np.column_stack([np.ones(4), t])
coef = np.linalg.solve(A6.T @ A6, A6.T @ y)
r = y - A6 @ coef
print(coef, (A6 @ coef).round(4), r.round(4))
print(np.isclose(r.sum(), 0), np.isclose(r @ t, 0), round(r @ r, 6))
```

```text
[1.5 0.8] [2.3 3.1 3.9 4.7] [-0.3 -0.1  1.1 -0.7]
True True 1.8
```

## 7. Overfitting

```python
V3 = np.vander(t, 4, increasing=True)
cubic = np.linalg.solve(V3, y)
print(cubic.round(4))
print("cubic at t = 5:", round(float(np.polyval(cubic[::-1], 5)), 4), "| line at t = 5:", round(float(coef[0] + coef[1] * 5), 4))
```

```text
[ 6.     -7.8333  4.5    -0.6667]
cubic at t = 5: -4.0 | line at t = 5: 5.5
```

The cubic is $6 - 7.83\,t + 4.5\,t^2 - 0.67\,t^3$. It matches the four data points exactly, but at $t = 5$ it predicts $-4$, while the line predicts $5.5$, close to the trend in the data. The data rise slowly with some scatter, so the line is much more believable. The cubic has as many free numbers as data points, so it can bend to fit the scatter exactly, and then it behaves wildly outside the points (overfitting).

## 8. Dependent columns

Check: $X(2,2,0) = (2 + 4,\ 4 + 0,\ 0 + 2,\ 6 + 2) = (6, 4, 2, 8)$. The nullspace of $X$ is spanned by $(1, 1, -1)$ (column 1 plus column 2 minus column 3 is zero), so every solution is $(2,2,0) + t\,(1,1,-1) = (2 + t,\ 2 + t,\ -t)$. The squared length is $2(2 + t)^2 + t^2 = 8 + 8t + 3t^2$, which is smallest at $t = -\tfrac43$ (set $8 + 6t = 0$). That gives $(\tfrac23, \tfrac23, \tfrac43)$, the minimum-norm solution. NumPy's `lstsq` returns it (and so does the pseudoinverse).

```python
X = np.array([[1, 2, 3], [2, 0, 2], [0, 1, 1], [3, 1, 4.0]])
yy = np.array([6.0, 4.0, 2.0, 8.0])
print(X @ np.array([2.0, 2.0, 0.0]))
w = np.linalg.lstsq(X, yy, rcond=None)[0]
print(w, np.allclose(X @ w, yy), np.isclose(np.linalg.det(X.T @ X), 0))
```

```text
[6. 4. 2. 8.]
[0.66666667 0.66666667 1.33333333] True True
```

## 9. Remove a direction

(a) $\mathbf{x}\cdot\mathbf{d} = 7/\sqrt2$, so the component along $\mathbf{d}$ is $(7/\sqrt2)\,\mathbf{d} = 3.5\,(1,1)$. Subtracting gives $(3,4) - (3.5, 3.5) = (-0.5, 0.5)$. Check: $(-0.5, 0.5)\cdot(1,1) = 0$.

(b) $`\mathbf{d}\mathbf{d}^\top = \tfrac12\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}`$, so
```math
I - \mathbf{d}\mathbf{d}^\top = \begin{bmatrix} 0.5 & -0.5 \\ -0.5 & 0.5 \end{bmatrix}.
```
Applying it to $(3,4)$ gives $(1.5 - 2,\ -1.5 + 2) = (-0.5, 0.5)$.

```python
d = np.array([1.0, 1.0]) / np.sqrt(2)
x = np.array([3.0, 4.0])
removed = x - (x @ d) * d
M = np.eye(2) - np.outer(d, d)
print(removed, np.isclose(removed @ d, 0))
print(M, np.allclose(M @ x, removed))
```

```text
[-0.5  0.5] True
[[ 0.5 -0.5]
 [-0.5  0.5]] True
```

## 10. Write and test

```python
import scipy.linalg as sla


def least_squares(A, b):
    """Solve min ||A x - b||^2 with QR: R x = Q^T b by back substitution."""
    Q, R = np.linalg.qr(A)
    return sla.solve_triangular(R, Q.T @ b)


# (a) a random 50 x 4 problem with noisy targets
rng = np.random.default_rng(0)
A = rng.normal(size=(50, 4))
b = A @ np.array([1.0, -2.0, 0.5, 3.0]) + rng.normal(0, 0.3, size=50)
x_qr = least_squares(A, b)
x_ref = np.linalg.lstsq(A, b, rcond=None)[0]
print(x_qr.round(4), np.allclose(x_qr, x_ref))
print("A^T r is zero:", np.allclose(A.T @ (b - A @ x_qr), 0))

# (b) polynomial degrees 1 to 8 on noisy samples of a sine wave
t_train = np.linspace(0, 1, 15)
y_train = np.sin(2 * np.pi * t_train) + rng.normal(0, 0.25, size=15)
t_test = np.linspace(0, 1, 200) + 0.002
y_test = np.sin(2 * np.pi * t_test) + rng.normal(0, 0.25, size=200)
print("degree  train RMSE  test RMSE")
for degree in range(1, 9):
    V = np.vander(t_train, degree + 1, increasing=True)
    coef = np.linalg.lstsq(V, y_train, rcond=None)[0]
    train = np.sqrt(np.mean((V @ coef - y_train) ** 2))
    test = np.sqrt(np.mean((np.vander(t_test, degree + 1, increasing=True) @ coef - y_test) ** 2))
    print(f"{degree:>6} {train:>11.3f} {test:>10.3f}")
```

```text
[ 1.0041 -2.0032  0.4991  3.0728] True
A^T r is zero: True
degree  train RMSE  test RMSE
     1       0.545      0.514
     2       0.544      0.515
     3       0.212      0.286
     4       0.209      0.288
     5       0.207      0.278
     6       0.205      0.278
     7       0.203      0.281
     8       0.193      0.297
```

The QR solution agrees with `lstsq`, and the residual is perpendicular to every column. In (b) the training error keeps falling as the degree rises (from 0.545 to 0.193), while the test error stops improving after the cubic, reaches its lowest value (0.278) at degrees 5 and 6 in this run, and is worse again at degree 8. The exact numbers depend on the random draw, but the pattern does not: training error only goes down, and test error has a valley.
