# Solutions: Linear Transformations

Worked answers to the [problems](problems.md). The code outputs were produced by running the code. Numerical checks hold only up to rounding error, so the code compares with `np.allclose`. Back to the [topic](../README.md).

## 1. Linear or not?

(a) Linear. In the first coordinate $x - y$, and in the second $3x$, every entry is a plain sum of multiples of the inputs, so both rules hold. The matrix has rows $(1, -1)$ and $(3, 0)$.

(b) Not linear: $T(0, 0) = (1, 0) \ne (0, 0)$. It is a shift.

(c) Not linear: $T(2, 2) = (4, 2)$ but $2\,T(1, 1) = 2\,(1, 1) = (2, 2)$. The product $xy$ breaks scaling.

(d) Linear, from $\mathbb{R}^3$ to $\mathbb{R}^2$. Its matrix has rows $(0, 0, 1)$ and $(1, 1, 1)$, and it is $2 \times 3$.

```python
import numpy as np

maps = {
    "(a)": lambda v: np.array([v[0] - v[1], 3 * v[0]]),
    "(b)": lambda v: np.array([v[0] + 1, v[1]]),
    "(c)": lambda v: np.array([v[0] * v[1], v[0]]),
}
rng = np.random.default_rng(0)
for name, T in maps.items():
    ok = True
    for _ in range(50):
        u, v, c = rng.normal(size=2), rng.normal(size=2), rng.normal()
        ok &= np.allclose(T(u + v), T(u) + T(v)) and np.allclose(T(c * v), c * T(v))
    print(name, "passes both rules:", bool(ok), "| T(0) =", T(np.zeros(2)))

T_d = lambda v: np.array([v[2], v[0] + v[1] + v[2]])
ok = True
for _ in range(50):
    u, v, c = rng.normal(size=3), rng.normal(size=3), rng.normal()
    ok &= np.allclose(T_d(u + v), T_d(u) + T_d(v)) and np.allclose(T_d(c * v), c * T_d(v))
print("(d) passes both rules:", bool(ok), "| matrix:\n", np.column_stack([T_d(e) for e in np.eye(3)]))
```

```text
(a) passes both rules: True | T(0) = [0. 0.]
(b) passes both rules: False | T(0) = [1. 0.]
(c) passes both rules: False | T(0) = [0. 0.]
(d) passes both rules: True | matrix:
 [[0. 0. 1.]
 [1. 1. 1.]]
```

## 2. Use linearity

Since $(1, 0) = \tfrac12[(1, 1) + (1, -1)]$ and $(0, 1) = \tfrac12[(1, 1) - (1, -1)]$, linearity gives

$$
T(1, 0) = \tfrac12[(3, 1) + (1, 5)] = (2, 3), \qquad T(0, 1) = \tfrac12[(3, 1) - (1, 5)] = (1, -2).
$$

So the matrix has columns $(2, 3)$ and $(1, -2)$, that is, rows $(2, 1)$ and $(3, -2)$. Then $T(5, -2) = 5\,(2, 3) - 2\,(1, -2) = (10 - 2,\ 15 + 4) = (8, 19)$.

```python
A = np.array([[2.0, 1.0],
              [3.0, -2.0]])
print(A @ [1, 1], A @ [1, -1])      # the two given values
print(A @ [5, -2])
```

```text
[3. 1.] [1. 5.]
[ 8. 19.]
```

## 3. Matrices from pictures

(a) Reflecting in the line $y = -x$ sends $(1, 0)$ to $(0, -1)$ and $(0, 1)$ to $(-1, 0)$, so the matrix has columns $(0, -1)$ and $(-1, 0)$.

(b) Rotation by $45^\circ$ has columns $(\cos 45^\circ, \sin 45^\circ) = (\tfrac{\sqrt2}{2}, \tfrac{\sqrt2}{2})$ and $(-\sin 45^\circ, \cos 45^\circ) = (-\tfrac{\sqrt2}{2}, \tfrac{\sqrt2}{2})$.

(c) $T(\mathbf{e}_1) = (1, \tfrac12)$ and $T(\mathbf{e}_2) = (0, 1)$, so the columns are $(1, 0.5)$ and $(0, 1)$.

```math
\text{(a)}\ \begin{bmatrix} 0 & -1 \\ -1 & 0 \end{bmatrix}, \qquad
\text{(b)}\ \frac{\sqrt2}{2}\begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}, \qquad
\text{(c)}\ \begin{bmatrix} 1 & 0 \\ 0.5 & 1 \end{bmatrix}.
```

```python
def matrix_from_images(T):
    return np.column_stack([T(e) for e in np.eye(2)])

refl = lambda v: np.array([-v[1], -v[0]])            # reflection in y = -x
a = np.deg2rad(45)
rot45 = lambda v: np.array([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]])
shear = lambda v: np.array([v[0], v[1] + v[0] / 2])
for name, T in [("(a)", refl), ("(b)", rot45), ("(c)", shear)]:
    print(name)
    print(matrix_from_images(T) + 0.0)
print(np.allclose(matrix_from_images(rot45), np.sqrt(2) / 2 * np.array([[1, -1], [1, 1]])))
```

```text
(a)
[[ 0. -1.]
 [-1.  0.]]
(b)
[[ 0.70710678 -0.70710678]
 [ 0.70710678  0.70710678]]
(c)
[[1.  0. ]
 [0.5 1. ]]
True
```

## 4. Order matters

With $R$ having rows $(0, -1)$ and $(1, 0)$ and $P$ having rows $(1, 0)$ and $(0, 0)$,

```math
PR = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & -1 \\ 0 & 0 \end{bmatrix}, \qquad
RP = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 0 \\ 1 & 0 \end{bmatrix}.
```

$PR$ means "turn, then flatten onto the $x$-axis". It sends $(2, 3)$ to $R(2,3) = (-3, 2)$ and then to $(-3, 0)$. $RP$ means "flatten onto the $x$-axis, then turn": $(2, 3) \mapsto (2, 0) \mapsto (0, 2)$. The results differ, so $PR \ne RP$.

Kernel and range: $PR$ sends $(x, y)$ to $(-y, 0)$, so its kernel is the $x$-axis ($y = 0$) and its range is the $x$-axis. $RP$ sends $(x, y)$ to $(0, x)$, so its kernel is the $y$-axis and its range is the $y$-axis. Both have rank 1, and $1 + 1 = 2$ in both cases.

```python
R = np.array([[0.0, -1.0], [1.0, 0.0]])
P = np.array([[1.0, 0.0], [0.0, 0.0]])
print("PR =\n", P @ R)
print("RP =\n", R @ P)
v = np.array([2.0, 3.0])
print("PR v =", (P @ R) @ v, "| RP v =", (R @ P) @ v)
print("ranks:", np.linalg.matrix_rank(P @ R), np.linalg.matrix_rank(R @ P))
```

```text
PR =
 [[ 0. -1.]
 [ 0.  0.]]
RP =
 [[0. 0.]
 [1. 0.]]
PR v = [-3.  0.] | RP v = [0. 2.]
ranks: 1 1
```

## 5. Kernel and range

Row reduce: row 3 equals row 1 plus row 2, so the third row becomes zero and there are two pivots, in columns 1 and 2. Column 3 is free. Solving $A\mathbf{x} = \mathbf{0}$ with $x_3 = 1$ gives $x_2 = -1$ and then $x_1 = -2x_2 - 3x_3 = -1$. So the kernel has the basis $(-1, -1, 1)$, of dimension 1. The range is spanned by the pivot columns $(1, 0, 1)$ and $(2, 1, 3)$ (the third column is their sum), a plane of dimension 2. Check: $1 + 2 = 3$.

$T$ is not invertible, because the kernel is not $\{\mathbf{0}\}$. The left nullspace relation row 3 = row 1 + row 2 says that every output satisfies $b_3 = b_1 + b_2$. For $(1, 1, 2)$ we have $2 = 1 + 1$, so it is in the range. For $(1, 1, 3)$ we have $3 \ne 2$, so it is not.

```python
from scipy.linalg import null_space

A = np.array([[1.0, 2.0, 3.0],
              [0.0, 1.0, 1.0],
              [1.0, 3.0, 4.0]])
K = null_space(A)
print("rank =", np.linalg.matrix_rank(A), "| kernel direction:", (K[:, 0] / K[2, 0]).round(4), "(scaled so the last entry is 1)")
print("A @ (-1, -1, 1) =", A @ [-1, -1, 1])
for b in ([1, 1, 2], [1, 1, 3]):
    x, res, rank, sv = np.linalg.lstsq(A, np.array(b, float), rcond=None)
    print(b, "in the range:", np.allclose(A @ x, b))
```

```text
rank = 2 | kernel direction: [-1. -1.  1.] (scaled so the last entry is 1)
A @ (-1, -1, 1) = [0. 0. 0.]
[1, 1, 2] in the range: True
[1, 1, 3] in the range: False
```

## 6. Differentiate and integrate

(a) $J$ sends $1 \mapsto x$, $x \mapsto \tfrac12x^2$, $x^2 \mapsto \tfrac13x^3$. In coordinates $(a_0, a_1, a_2, a_3)$ these are the columns $(0, 1, 0, 0)$, $(0, 0, \tfrac12, 0)$, $(0, 0, 0, \tfrac13)$. Applying $D$ to $1, x, x^2, x^3$ gives $0, 1, 2x, 3x^2$, which in $\mathcal{P}_2$ coordinates are the columns $(0, 0, 0)$, $(1, 0, 0)$, $(0, 2, 0)$, $(0, 0, 3)$.

```math
J = \begin{bmatrix} 0 & 0 & 0 \\ 1 & 0 & 0 \\ 0 & \tfrac12 & 0 \\ 0 & 0 & \tfrac13 \end{bmatrix}, \qquad
D = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 2 & 0 \\ 0 & 0 & 0 & 3 \end{bmatrix}.
```

(b) $DJ$ is $3 \times 3$ and equals the identity: differentiating an integral returns the original polynomial. $JD$ is $4 \times 4$ and equals $\operatorname{diag}(0, 1, 1, 1)$: integrating a derivative returns the polynomial with its constant term replaced by $0$. The constant is in the kernel of $D$, and no later step can restore what $D$ erased.

```python
J = np.array([[0, 0, 0],
              [1, 0, 0],
              [0, 1 / 2, 0],
              [0, 0, 1 / 3]])
D = np.array([[0, 1, 0, 0],
              [0, 0, 2, 0],
              [0, 0, 0, 3]], dtype=float)
print("D J =\n", D @ J)
print("J D =\n", J @ D)
```

```text
D J =
 [[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
J D =
 [[0. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
```

## 7. A diagonalizing basis

(a) $A\mathbf{m}_1 = (4 + 1,\ 2 + 3) = (5, 5) = 5\,\mathbf{m}_1$ and $A\mathbf{m}_2 = (4 - 2,\ 2 - 6) = (2, -4) = 2\,\mathbf{m}_2$. In the new basis $\mathbf{m}_1$ is stretched by 5 and $\mathbf{m}_2$ by 2, so $B = \operatorname{diag}(5, 2)$.

(b) Here $M$ has rows $(1, 1)$ and $(1, -2)$, with $\det M = -3$ and $`M^{-1} = \tfrac13\begin{bmatrix} 2 & 1 \\ 1 & -1 \end{bmatrix}`$. Then $A^3 = M\,\operatorname{diag}(125, 8)\,M^{-1}$ and

```math
A^3 = \begin{bmatrix} 125 & 8 \\ 125 & -16 \end{bmatrix}\cdot\frac13\begin{bmatrix} 2 & 1 \\ 1 & -1 \end{bmatrix} = \frac13\begin{bmatrix} 258 & 117 \\ 234 & 141 \end{bmatrix} = \begin{bmatrix} 86 & 39 \\ 78 & 47 \end{bmatrix}.
```

```python
A = np.array([[4.0, 1.0], [2.0, 3.0]])
M = np.array([[1.0, 1.0], [1.0, -2.0]])
B = np.linalg.inv(M) @ A @ M
print("B =\n", B.round(10) + 0.0)
A3 = M @ np.diag([5.0**3, 2.0**3]) @ np.linalg.inv(M)
print("A^3 via the new basis =\n", A3.round(10))
print(np.allclose(A3, A @ A @ A), np.allclose(A3, np.linalg.matrix_power(A, 3)))
```

```text
B =
 [[5. 0.]
 [0. 2.]]
A^3 via the new basis =
 [[86. 39.]
 [78. 47.]]
True True
```

## 8. Similar matrices

(a) $M^{-1}$ has rows $(1, -1)$ and $(0, 1)$. Then $AM$ has rows $(1, 3)$ and $(3, 7)$, and $B = M^{-1}(AM)$ has rows $(1 - 3,\ 3 - 7) = (-2, -4)$ and $(3, 7)$. The traces are $1 + 4 = 5$ and $-2 + 7 = 5$. The determinants are $1\cdot4 - 2\cdot3 = -2$ and $(-2)(7) - (-4)(3) = -14 + 12 = -2$. They match, as they must for similar matrices.

(b) For any invertible $M$, $M^{-1}IM = M^{-1}M = I$. So every matrix similar to $I$ is $I$ itself, and the shear is not similar to $I$ even though their traces (both 2) and determinants (both 1) agree. Equal trace and determinant are necessary for being similar, but not sufficient. The shear is a different transformation: it is not the identity in any basis.

```python
A = np.array([[1.0, 2.0], [3.0, 4.0]])
M = np.array([[1.0, 1.0], [0.0, 1.0]])
B = np.linalg.inv(M) @ A @ M
print("B =\n", B)
print("trace:", np.trace(A), np.trace(B), "| det:", round(np.linalg.det(A), 6), round(np.linalg.det(B), 6))

S = np.array([[1.0, 1.0], [0.0, 1.0]])
M2 = np.array([[2.0, 1.0], [5.0, 3.0]])               # an arbitrary invertible matrix
print("M^-1 I M = I:", np.allclose(np.linalg.inv(M2) @ np.eye(2) @ M2, np.eye(2)))
print("trace and det of I and S:", np.trace(np.eye(2)), np.trace(S), np.linalg.det(np.eye(2)), np.linalg.det(S))
```

```text
B =
 [[-2. -4.]
 [ 3.  7.]]
trace: 5.0 5.0 | det: -2.0 -2.0
M^-1 I M = I: True
trace and det of I and S: 2.0 2.0 1.0 1.0
```

## 9. Scale about a point

Shift by $(-1, -1)$ so that $(1, 1)$ goes to the origin, scale by 2, and shift by $(1, 1)$. In homogeneous coordinates, with the first step nearest the point,

```math
H = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 1 \end{bmatrix}
\begin{bmatrix} 2 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 1 \end{bmatrix}
\begin{bmatrix} 1 & 0 & -1 \\ 0 & 1 & -1 \\ 0 & 0 & 1 \end{bmatrix}
= \begin{bmatrix} 2 & 0 & -1 \\ 0 & 2 & -1 \\ 0 & 0 & 1 \end{bmatrix}.
```

Applied to $(3, 1, 1)$ it gives $(6 - 1,\ 2 - 1,\ 1) = (5, 1, 1)$: the point $(3, 1)$ was $2$ units right of $(1, 1)$ on the same row, and now it is $4$ units right. For $(1, 1, 1)$ we get $(2 - 1,\ 2 - 1,\ 1) = (1, 1, 1)$, so the center stays put.

```python
def shift(b):
    H = np.eye(3); H[:2, 2] = b; return H

scale2 = np.diag([2.0, 2.0, 1.0])
H = shift((1, 1)) @ scale2 @ shift((-1, -1))
print(H)
print(H @ [3, 1, 1], H @ [1, 1, 1])
```

```text
[[ 2.  0. -1.]
 [ 0.  2. -1.]
 [ 0.  0.  1.]]
[5. 1. 1.] [1. 1. 1.]
```

## 10. Write and test

`matrix_of` loops over the standard basis vectors, applies `T` to each, and stacks the answers as columns. A linear map is rebuilt exactly. For a nonlinear map the matrix is only a guess, because the images of the basis vectors do not determine the map, and checking it on other inputs exposes the mismatch.

```python
def matrix_of(T, n):
    return np.column_stack([T(e) for e in np.eye(n)])

# (a) a rotation by 25 degrees, and a hidden random matrix
a = np.deg2rad(25)
rot = lambda v: np.array([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]])
print(np.allclose(matrix_of(rot, 2), [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]))
rng = np.random.default_rng(5)
hidden = rng.normal(size=(3, 4))
print("hidden matrix recovered:", np.allclose(matrix_of(lambda v: hidden @ v, 4), hidden))

# (b) convolution with the kernel (1, 2, 3), "same" size, length 6
conv = lambda x: np.convolve(x, [1.0, 2.0, 3.0], mode="same")
C = matrix_of(conv, 6)
print(C)
x = rng.normal(size=6)
print("matrix reproduces the convolution:", np.allclose(C @ x, conv(x)))

# (c) a nonlinear map
sq = lambda x: x ** 2
Q = matrix_of(sq, 6)
print("matrix reproduces the squaring map:", np.allclose(Q @ x, sq(x)))
```

```text
True
hidden matrix recovered: True
[[2. 1. 0. 0. 0. 0.]
 [3. 2. 1. 0. 0. 0.]
 [0. 3. 2. 1. 0. 0.]
 [0. 0. 3. 2. 1. 0.]
 [0. 0. 0. 3. 2. 1.]
 [0. 0. 0. 0. 3. 2.]]
matrix reproduces the convolution: True
matrix reproduces the squaring map: False
```

For (b), the matrix has the same three numbers repeating along each diagonal, $(3, 2, 1)$ read along a row, with the pattern clipped at the two ends. It is a banded Toeplitz matrix, which is why a convolution layer has few weights even though it is a linear map on a long vector. For (c), $Q$ turns out to be the identity matrix (since $1^2 = 1$ and $0^2 = 0$), which certainly does not square its input: a nonlinear map is not determined by its values on the basis vectors.
