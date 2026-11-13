# Solutions: Singular Value Decomposition

Worked answers to the [problems](problems.md). The code outputs were produced by running the code, and the blocks run top to bottom in one session (later blocks reuse names from earlier ones, such as `np`). Numerical checks hold only up to rounding error, so the code compares with `np.allclose`. Signs of singular vectors can differ from yours, because flipping a pair $(\mathbf{u}_i, \mathbf{v}_i)$ gives an equally valid SVD. Back to the [topic](../README.md).

## 1. Singular values from geometry

For $A$: $`A^\top A = \begin{bmatrix} 9 & 0 \\ 0 & 4 \end{bmatrix}`$ has eigenvalues $9$ and $4$, so $\sigma_1 = 3$ and $\sigma_2 = 2$. The ellipse has half-axes $3$ and $2$ (the $-2$ only flips the second axis), and $\sigma_1\sigma_2 = 6 = |\det A|$.

For $B$: $`B^\top B = \begin{bmatrix} 1 & 0 \\ 0 & 4 \end{bmatrix}`$ has eigenvalues $4$ and $1$ (sorted), so $\sigma_1 = 2$ and $\sigma_2 = 1$. The half-axes are $2$ and $1$, and $\sigma_1\sigma_2 = 2 = |\det B|$. Here $B$ turns the picture by $90^\circ$ as well, which is why $U$ and $V$ are not the identity even though the singular values are easy.

```python
import numpy as np

A = np.array([[3.0, 0.0], [0.0, -2.0]])
B = np.array([[0.0, -2.0], [1.0, 0.0]])
for name, M in [("A", A), ("B", B)]:
    print(name, np.linalg.svd(M, compute_uv=False), abs(np.linalg.det(M)))
```

```text
A [3. 2.] 6.0
B [2. 1.] 2.0
```

## 2. A full SVD by hand

First $`A^\top A = \begin{bmatrix} 5 & 3 \\ 3 & 5 \end{bmatrix}`$. Its characteristic equation is $\lambda^2 - 10\lambda + 16 = 0$, so $\lambda = 8$ and $\lambda = 2$, with unit eigenvectors $\mathbf{v}_1 = (1,1)/\sqrt2$ and $\mathbf{v}_2 = (1,-1)/\sqrt2$. Hence $\sigma_1 = \sqrt8 = 2\sqrt2$ and $\sigma_2 = \sqrt2$.

Next, $A\mathbf{v}_1 = (4, 0)/\sqrt2 = (2\sqrt2, 0)$, so $\mathbf{u}_1 = (1, 0)$. And $A\mathbf{v}_2 = (0, -2)/\sqrt2 = (0, -\sqrt2)$, so $\mathbf{u}_2 = (0, -1)$. Therefore

```math
A = \underbrace{\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}}_{U}
\underbrace{\begin{bmatrix} 2\sqrt2 & 0 \\ 0 & \sqrt2 \end{bmatrix}}_{\Sigma}
\underbrace{\frac{1}{\sqrt2}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}}_{V^\top},
\qquad |\det A| = \sigma_1\sigma_2 = 4 .
```

```python
A = np.array([[2.0, 2.0], [-1.0, 1.0]])
U_hand = np.array([[1.0, 0.0], [0.0, -1.0]])
S_hand = np.diag([2 * np.sqrt(2), np.sqrt(2)])
Vt_hand = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
print(np.allclose(U_hand @ S_hand @ Vt_hand, A))
print(np.linalg.eigvalsh(A.T @ A))
print(np.linalg.svd(A, compute_uv=False), abs(np.linalg.det(A)))
```

```text
True
[2. 8.]
[2.82842712 1.41421356] 4.0
```

## 3. A rank-one matrix and its subspaces

(a) Every row is a multiple of $(1,1,1)$, so the rank is $1$. Writing $C = (1,2)^\top(1,1,1)$, the one nonzero singular value is the product of the lengths: $\sigma_1 = \sqrt5\cdot\sqrt3 = \sqrt{15} \approx 3.873$.

(b) $\mathbf{u}_1 = (1,2)/\sqrt5$ and $\mathbf{v}_1 = (1,1,1)/\sqrt3$, so that $C = \sigma_1\mathbf{u}_1\mathbf{v}_1^\top$.

(c) The column space is the line through $(1,2)$ in $\mathbb{R}^2$ (spanned by $\mathbf{u}_1$). The left nullspace is the perpendicular line through $(2,-1)$, which is $\mathbf{u}_2$. The row space is the line through $(1,1,1)$ in $\mathbb{R}^3$ (spanned by $\mathbf{v}_1$). The nullspace is the plane $x_1 + x_2 + x_3 = 0$, of dimension $2$, and $\mathbf{v}_2, \mathbf{v}_3$ are any orthonormal basis of it, for example $(1,-1,0)/\sqrt2$ and $(1,1,-2)/\sqrt6$.

```python
C = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
U, s, Vt = np.linalg.svd(C)
print(np.linalg.matrix_rank(C), s.round(4), np.sqrt(15).round(4))
r = 1
print(np.allclose(C @ Vt[r:].T, 0), np.allclose(U[:, r:].T @ C, 0))
v2 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
v3 = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)
print(np.allclose(C @ v2, 0), np.allclose(C @ v3, 0), round(v2 @ v3, 12))
```

```text
1 [3.873 0.   ] 3.873
True True
True True 0.0
```

## 4. Reading the singular values

(a) Three singular values are nonzero, so $r = 3$. (b) The column space and row space have dimension $3$: bases $\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3$ and $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$. The nullspace has dimension $4 - 3 = 1$ (basis $\mathbf{v}_4$), and the left nullspace has dimension $5 - 3 = 2$ (basis $\mathbf{u}_4, \mathbf{u}_5$). (c) $\kappa = 7/0.5 = 14$. (d) The energy is $\sigma_1^2 + \sigma_2^2 + \sigma_3^2 = 49 + 9 + 0.25 = 58.25$, and rank $2$ keeps $58/58.25 \approx 99.57\%$. The 2-norm error is $\sigma_3 = 0.5$.

```python
s = np.array([7.0, 3.0, 0.5, 0.0])
print(int(np.sum(s > 1e-12)), s[0] / s[2])
print(np.sum(s[:2] ** 2) / np.sum(s ** 2), s[2])
```

```text
3 14.0
0.9957081545064378 0.5
```

## 5. U and V from the two Gram matrices

(a) $`A^\top A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}`$ has eigenvalues $3$ and $1$. The $3\times3$ matrix $`AA^\top = \begin{bmatrix} 1 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 1 \end{bmatrix}`$ has eigenvalues $3$, $1$, and $0$. They share the nonzero eigenvalues $3$ and $1$, and $AA^\top$ has one extra eigenvalue $0$ because it is $3\times3$ but has rank $2$. (b) $\sigma_1 = \sqrt3$ and $\sigma_2 = 1$. (c) The left nullspace is spanned by the eigenvector of $AA^\top$ with eigenvalue $0$, which is $(1,-1,1)/\sqrt3$. Check: $A^\top(1,-1,1) = (1 - 1 + 0,\ 0 - 1 + 1) = (0, 0)$.

```python
A = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
print(np.linalg.eigvalsh(A.T @ A), np.linalg.eigvalsh(A @ A.T).round(10) + 0.0)
print(np.linalg.svd(A, compute_uv=False), np.sqrt(3).round(4))
print(A.T @ np.array([1.0, -1.0, 1.0]))
```

```text
[1. 3.] [0. 1. 3.]
[1.73205081 1.        ] 1.7321
[0. 0.]
```

## 6. Pseudoinverse and minimum norm

(a) $B = \sigma_1\mathbf{u}_1\mathbf{v}_1^\top$, so $B^+ = \frac{1}{\sigma_1}\mathbf{v}_1\mathbf{u}_1^\top = \frac{1}{70}(1,2)^\top(1,2,3)$, which equals $B^\top/70$.

(b) $\mathbf{x}^+ = B^+\mathbf{b} = \frac{1}{70}(1 + 4 + 12)\,(1,2) = \frac{17}{70}(1,2) \approx (0.243,\ 0.486)$.

(c) The nullspace is spanned by $(2,-1)$, so all least-squares solutions are $\mathbf{x}^+ + t(2,-1)$. Because $\mathbf{x}^+$ is a multiple of $(1,2)$, it is perpendicular to $(2,-1)$ (the dot product is $2 - 2 = 0$), so $\|\mathbf{x}^+ + t(2,-1)\|^2 = \|\mathbf{x}^+\|^2 + 5t^2$, smallest at $t = 0$.

```python
B = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
b = np.array([1.0, 2.0, 4.0])
B_plus = np.linalg.pinv(B)
print(np.allclose(B_plus, B.T / 70))
x_plus = B_plus @ b
print(x_plus, 17 / 70 * np.array([1, 2]))
print(np.allclose(x_plus, np.linalg.lstsq(B, b, rcond=None)[0]))
n = np.array([2.0, -1.0])
for t in (-0.2, 0.0, 0.2):
    print(t, np.linalg.norm(x_plus + t * n).round(4))
print(np.allclose(B.T @ (b - B @ x_plus), 0))
```

```text
True
[0.24285714 0.48571429] [0.24285714 0.48571429]
True
-0.2 0.7035
0.0 0.543
0.2 0.7035
True
```

## 7. A nearly singular system

(a) Subtracting the first equation from the second gives $0.01\,x_2 = 0.01$ for the first right-hand side, so $x_2 = 1$ and $\mathbf{x} = (1, 1)$. For the second right-hand side, $0.01\,x_2 = 0.02$, so $x_2 = 2$, $x_1 = 0$, and $\mathbf{x} = (0, 2)$.

(b) $|\det A| = 0.01$ and the sum of squared entries is $1 + 1 + 1 + 1.0201 = 4.0201$. With $\sigma_1^2 + \sigma_2^2 = 4.0201$ and $\sigma_1\sigma_2 = 0.01$, the large one is about $\sigma_1 \approx 2.005$, so $\sigma_2 \approx 0.01/2.005 \approx 0.00499$ and $\kappa \approx 402$.

(c) The right-hand side changes from $(2, 2.01)$ to $(2, 2.02)$: a relative change of $0.01/2.84 \approx 0.35\%$. The solution changes from $(1,1)$ to $(0,2)$: a relative change of $\sqrt2/\sqrt2 = 100\%$. The magnification is about $283$, below $\kappa \approx 402$.

```python
A = np.array([[1.0, 1.0], [1.0, 1.01]])
b1, b2 = np.array([2.0, 2.01]), np.array([2.0, 2.02])
x1, x2 = np.linalg.solve(A, b1), np.linalg.solve(A, b2)
print(x1, x2)
s = np.linalg.svd(A, compute_uv=False)
print(s, s[0] / s[1], np.linalg.cond(A))
rel_b = np.linalg.norm(b2 - b1) / np.linalg.norm(b1)
rel_x = np.linalg.norm(x2 - x1) / np.linalg.norm(x1)
print(rel_b, rel_x, rel_x / rel_b)
```

```text
[1. 1.] [0. 2.]
[2.0050125 0.0049875] 402.0075124842952 402.0075124842952
0.003526706147109111 1.0000000000000222 283.55070093371285
```

## 8. Best rank-one approximation

$A_1 = 2\sqrt2\,(1,0)^\top\frac{(1,1)}{\sqrt2} = (2, 0)^\top(1,1)$, that is

```math
A_1 = \begin{bmatrix} 2 & 2 \\ 0 & 0 \end{bmatrix}, \qquad A - A_1 = \begin{bmatrix} 0 & 0 \\ -1 & 1 \end{bmatrix} .
```

The difference is the second layer, $\sigma_2\mathbf{u}_2\mathbf{v}_2^\top$, which has rank one, so its 2-norm and Frobenius norm are both $\sigma_2 = \sqrt2 \approx 1.414$. The energy kept is $\sigma_1^2/(\sigma_1^2 + \sigma_2^2) = 8/10 = 80\%$.

```python
A = np.array([[2.0, 2.0], [-1.0, 1.0]])
U, s, Vt = np.linalg.svd(A)
A1 = s[0] * np.outer(U[:, 0], Vt[0])
print(A1.round(6) + 0.0)
E = A - A1
print(E.round(6) + 0.0)
print(np.linalg.norm(E, 2), np.linalg.norm(E, "fro"), s[1])
print(s[0] ** 2 / np.sum(s ** 2))
```

```text
[[2. 2.]
 [0. 0.]]
[[ 0.  0.]
 [-1.  1.]]
1.4142135623730951 1.4142135623730951 1.4142135623730951
0.7999999999999999
```

## 9. Storage and choosing k

(a) $k(m + n + 1) = k\cdot1601$ numbers, against $mn = 600{,}000$. Compression saves space while $k < 600000/1601 \approx 374.8$, that is, up to $k = 374$.

(b) The squared singular values are $2500\cdot0.64^{\,i-1}$, a geometric series, so the energy kept by $k$ terms is $1 - 0.64^{k}$ (the tail after $600$ terms is negligible). We need $0.64^k \le 0.01$, so $k \ge \ln 0.01/\ln 0.64 \approx 10.3$, and the smallest $k$ is $11$. It stores $11\cdot1601 = 17{,}611$ numbers, about $2.9\%$ of the original.

```python
m, n = 1000, 600
print(m * n / (m + n + 1))
sig = 50 * 0.8 ** np.arange(600)
energy = np.cumsum(sig ** 2) / np.sum(sig ** 2)
k = int(np.searchsorted(energy, 0.99) + 1)
print(k, energy[k - 2], energy[k - 1])
print(k * (m + n + 1), k * (m + n + 1) / (m * n))
```

```text
374.76577139287946
11 0.9884707849539314 0.992621302370516
17611 0.029351666666666668
```

## 10. PCA through the SVD (code)

Using the data above:

```python
rng = np.random.default_rng(10)
X = rng.normal(size=(150, 3)) @ np.array([[2.0, 0.5, 0.0], [0.0, 1.0, 0.3], [0.0, 0.0, 0.2]]) + np.array([8.0, -5.0, 3.0])
mean = X.mean(axis=0)
Xc = X - mean
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
w, V = np.linalg.eigh(np.cov(X.T))
print(np.allclose(s ** 2 / (len(X) - 1), w[::-1]), np.allclose(np.abs(Vt), np.abs(V[:, ::-1].T)))
print((s ** 2 / np.sum(s ** 2)).round(4))

scores1 = Xc @ Vt[:1].T                       # 150 x 1
X1 = scores1 @ Vt[:1] + mean                  # rank-one reconstruction, mean added back
print(np.isclose(np.sum((X - X1) ** 2), np.sum(s[1:] ** 2)), np.sum(s[1:] ** 2).round(3))

U0, s0, Vt0 = np.linalg.svd(X, full_matrices=False)       # without centering
print(Vt[0].round(3), Vt0[0].round(3))
print((mean / np.linalg.norm(mean)).round(3))
print(abs(Vt0[0] @ mean / np.linalg.norm(mean)).round(4))
```

```text
True True
[0.8328 0.1584 0.0088]
True 121.464
[0.955 0.296 0.021] [-0.811  0.509 -0.289]
[ 0.8   -0.522  0.295]
0.9998
```

(a) and (c) print `True`, so the SVD route reproduces the covariance eigen-decomposition, and the squared error of the rank-one reconstruction is exactly the sum of the dropped $\sigma_i^2$. (b) The first axis explains about $83\%$ of the variance and the second about $16\%$. (d) Without centering, the first right singular vector points almost along the mean vector $(8, -5, 3)$, up to sign (compare the second vector on the line of axes with the normalized mean vector printed below it; the absolute cosine of the angle between them is about $0.9998$), because the average position is much larger than the spread. It no longer describes the direction of greatest variation, which is why PCA always centers first.
