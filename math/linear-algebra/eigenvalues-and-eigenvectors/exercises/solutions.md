# Solutions: Eigenvalues and Eigenvectors

Worked answers to the [problems](problems.md). The code outputs were produced by running the code. Numerical checks hold only up to rounding error, so the code compares with `np.allclose`. Back to the [topic](../README.md).

## 1. Is it an eigenvector?

$A(1,2) = (4 - 2,\ 2 + 2) = (2, 4) = 2\,(1,2)$: yes, with eigenvalue $2$. $A(1,1) = (3, 3) = 3\,(1,1)$: yes, with eigenvalue $3$. $A(1,0) = (4, 2)$ is not a multiple of $(1,0)$: no.

```python
import numpy as np

A = np.array([[4.0, -1.0], [2.0, 1.0]])
for x in ([1, 2], [1, 1], [1, 0]):
    print(x, A @ np.array(x))
print(np.sort(np.linalg.eigvals(A)))
```

```text
[1, 2] [2. 4.]
[1, 1] [3. 3.]
[1, 0] [4. 2.]
[2. 3.]
```

## 2. The characteristic equation

$`\det\begin{bmatrix} 5-\lambda & 2 \\ 2 & 2-\lambda \end{bmatrix} = (5-\lambda)(2-\lambda) - 4 = \lambda^2 - 7\lambda + 6 = (\lambda - 6)(\lambda - 1)`$, so $\lambda = 6$ or $\lambda = 1$. For $\lambda = 6$, the matrix $A - 6I$ has rows $(-1, 2)$ and $(2, -4)$, and its nullspace is spanned by $(2, 1)$. For $\lambda = 1$, $A - I$ has rows $(4, 2)$ and $(2, 1)$, with nullspace spanned by $(1, -2)$. The dot product is $2\cdot1 + 1\cdot(-2) = 0$, so they are orthogonal. This was expected because $A$ is symmetric, and eigenvectors of a symmetric matrix for different eigenvalues are orthogonal.

```python
B = np.array([[5.0, 2.0], [2.0, 2.0]])
w, V = np.linalg.eigh(B)
print(w)
print(np.allclose(B @ np.array([2, 1]), 6 * np.array([2, 1])), np.allclose(B @ np.array([1, -2]), np.array([1, -2])))
print(np.array([2, 1]) @ np.array([1, -2]))
```

```text
[1. 6.]
True True
0
```

## 3. A triangular matrix

For a triangular matrix the eigenvalues are the diagonal entries: $3$, $5$, $1$. Check: their sum is $9 = 3 + 5 + 1$ (the trace), and their product is $15 = \det A$. No eigenvalue is $0$, so $A$ is invertible. The eigenvalues of $A^2$ are $9, 25, 1$. Those of $A^{-1}$ are $\tfrac13, \tfrac15, 1$. Those of $A + 2I$ are $5, 7, 3$.

```python
T = np.array([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0], [0.0, 0.0, 1.0]])
print(np.linalg.eigvals(T), np.trace(T), round(np.linalg.det(T), 6))
print(np.sort(np.linalg.eigvals(T @ T)), np.sort(np.linalg.eigvals(np.linalg.inv(T))).round(4), np.sort(np.linalg.eigvals(T + 2 * np.eye(3))))
```

```text
[3. 5. 1.] 9.0 15.0
[ 1.  9. 25.] [0.2    0.3333 1.    ] [3. 5. 7.]
```

## 4. Using the properties

$\det A = 1\cdot2\cdot4 = 8$ and $\operatorname{tr}A = 1 + 2 + 4 = 7$. The eigenvalues of $A^2 + I$ are $1^2 + 1 = 2$, $2^2 + 1 = 5$, and $4^2 + 1 = 17$. Those of $A^{-1}$ are $1$, $\tfrac12$, and $\tfrac14$. Those of $A + 3I$ are $4$, $5$, and $7$. $A$ is not singular, since no eigenvalue is $0$ (equivalently, $\det A = 8 \ne 0$).

## 5. Projections, reflections, rotations

(a) A vector on the line, such as $(3,4)$, is unchanged, so $\lambda = 1$. A vector perpendicular to the line, such as $(-4,3)$, is sent to $\mathbf{0}$, so $\lambda = 0$. Check: $P(3,4) = (1.08 + 1.92,\ 1.44 + 2.56) = (3, 4)$.

(b) $(1,0)$ is unchanged, so $\lambda = 1$. $(0,1)$ is flipped to $(0,-1)$, so $\lambda = -1$.

(c) A rotation by $90^\circ$ turns every nonzero vector to a perpendicular one, so no vector stays on its own line, and there is no real eigenvector. The characteristic equation is $\lambda^2 + 1 = 0$, so the eigenvalues are the complex numbers $i$ and $-i$.

```python
P = np.array([[0.36, 0.48], [0.48, 0.64]])
print(P @ np.array([3, 4]), P @ np.array([-4, 3]))
print(np.linalg.eigvalsh(P).round(6))
print(np.linalg.eigvals(np.array([[0.0, -1.0], [1.0, 0.0]])))
```

```text
[3. 4.] [0. 0.]
[0. 1.]
[0.+1.j 0.-1.j]
```

## 6. Diagonalize

The trace is $7$ and the determinant is $5\cdot2 - 4\cdot1 = 6$, so $\lambda^2 - 7\lambda + 6 = 0$ and $\lambda = 6$ or $\lambda = 1$. For $\lambda = 6$, $A - 6I$ has rows $(-1, 4)$ and $(1, -4)$, with nullspace spanned by $(4, 1)$. For $\lambda = 1$, $A - I$ has rows $(4, 4)$ and $(1, 1)$, with nullspace spanned by $(1, -1)$. So $S$ has columns $(4,1)$ and $(1,-1)$, and $\Lambda = \operatorname{diag}(6, 1)$. Then $A^3 = S\,\operatorname{diag}(216, 1)\,S^{-1}$, which gives the matrix with rows $(173, 172)$ and $(43, 44)$.

```python
A6 = np.array([[5.0, 4.0], [1.0, 2.0]])
S = np.array([[4.0, 1.0], [1.0, -1.0]])
Lam = np.diag([6.0, 1.0])
print(np.allclose(S @ Lam @ np.linalg.inv(S), A6))
A3 = S @ np.diag([6.0 ** 3, 1.0]) @ np.linalg.inv(S)
print(A3.round(6))
print(np.allclose(A3, np.linalg.matrix_power(A6, 3)))
```

```text
True
[[173. 172.]
 [ 43.  44.]]
True
```

## 7. A long-run state

The trace is $1.4$ and the determinant is $0.9\cdot0.5 - 0.5\cdot0.1 = 0.4$, so $\lambda^2 - 1.4\lambda + 0.4 = (\lambda - 1)(\lambda - 0.4)$: the eigenvalues are $1$ and $0.4$. For $\lambda = 1$, $M - I$ has rows $(-0.1, 0.5)$ and $(0.1, -0.5)$, with nullspace spanned by $(5, 1)$. Scaled to add to $1$: $(5/6, 1/6) \approx (0.833, 0.167)$. After many years the shares approach $(0.833, 0.167)$, whatever the start, because $M^k$ acts on the second eigenvector by $0.4^k$, which dies out, while the eigenvalue $1$ part stays. This stationary vector is a Markov chain's long-run distribution.

```python
M = np.array([[0.9, 0.5], [0.1, 0.5]])
print(np.sort(np.linalg.eigvals(M)))
print(np.linalg.matrix_power(M, 50) @ np.array([0.0, 1.0]))
print(np.allclose(M @ np.array([5 / 6, 1 / 6]), np.array([5 / 6, 1 / 6])))
```

```text
[0.4 1. ]
[0.83333333 0.16666667]
True
```

## 8. A defective matrix

The matrix $`\begin{bmatrix} 3 & 1 \\ 0 & 3 \end{bmatrix}`$ is triangular with $\lambda = 3$ twice. The matrix minus $3I$ has rows $(0, 1)$ and $(0, 0)$: rank $1$, so the nullspace has dimension $1$, spanned by $(1, 0)$. There is only one independent eigenvector, so the matrix cannot be diagonalized. The diagonal matrix with $3$ on the diagonal has the same eigenvalues, but subtracting $3I$ gives the zero matrix, whose nullspace is all of $\mathbb{R}^2$: two independent eigenvectors, so it is diagonalizable (it is already diagonal).

```python
D = np.array([[3.0, 1.0], [0.0, 3.0]])
print(np.linalg.matrix_rank(D - 3 * np.eye(2)), np.linalg.matrix_rank(np.zeros((2, 2))))
w, Sd = np.linalg.eig(D)
print(w, f"{np.linalg.cond(Sd):.1e}")
```

```text
1 0
[3. 3.] 3.0e+15
```

The huge condition number of the eigenvector matrix that NumPy returns for $D$ is the warning that its two columns are almost parallel.

## 9. The spectral theorem

The eigenvalues of $C$ are $3$ (eigenvector $(1,1)$) and $1$ (eigenvector $(1,-1)$). Normalizing, $\mathbf{q}_1 = (1,1)/\sqrt2$ and $\mathbf{q}_2 = (1,-1)/\sqrt2$, so $Q$ has those columns and $\Lambda = \operatorname{diag}(3, 1)$. Then $`\mathbf{q}_1\mathbf{q}_1^\top = \tfrac12\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}`$ and $`\mathbf{q}_2\mathbf{q}_2^\top = \tfrac12\begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}`$, and $3\mathbf{q}_1\mathbf{q}_1^\top + \mathbf{q}_2\mathbf{q}_2^\top$ has diagonal $\tfrac32 + \tfrac12 = 2$ and off-diagonal $\tfrac32 - \tfrac12 = 1$, which is $C$. Both eigenvalues are positive, so $C$ is positive definite. For $\mathbf{x} = (1,-1)$, $C\mathbf{x} = (1,-1)$ and $\mathbf{x}^\top C\mathbf{x} = 2 > 0$. The matrix with rows $(1,2)$ and $(2,1)$ has eigenvalues $3$ and $-1$, so it is not positive definite.

```python
C = np.array([[2.0, 1.0], [1.0, 2.0]])
w, Q = np.linalg.eigh(C)
print(w, np.allclose(Q.T @ Q, np.eye(2)))
pieces = sum(w[i] * np.outer(Q[:, i], Q[:, i]) for i in range(2))
print(np.allclose(pieces, C))
x = np.array([1.0, -1.0])
print(x @ C @ x, np.linalg.eigvalsh(np.array([[1.0, 2.0], [2.0, 1.0]])))
```

```text
[1. 3.] True
True
2.0 [-1.  3.]
```

## 10. Power method and step size

```python
def power_method(A, v0, steps):
    """Multiply by A and normalize, many times. Returns the vector and the Rayleigh estimate."""
    v = v0 / np.linalg.norm(v0)
    for _ in range(steps):
        v = A @ v
        v = v / np.linalg.norm(v)
    return v, v @ A @ v


A10 = np.array([[4.0, 1.0], [2.0, 3.0]])
true_vec = np.array([1.0, 1.0]) / np.sqrt(2)               # eigenvector for lambda = 5
errors = []
for steps in range(1, 9):
    v, estimate = power_method(A10, np.array([1.0, 0.0]), steps)
    errors.append(np.sqrt(max(0.0, 1 - (v @ true_vec) ** 2)))
print(np.round(errors, 5))
print("ratios of consecutive errors:", np.round(np.array(errors[1:]) / np.array(errors[:-1]), 3))
print("Rayleigh estimate after 8 steps:", round(estimate, 6))

H = np.diag([1.0, 10.0])
for alpha in (0.19, 0.21):
    x = np.array([1.0, 1.0])
    for _ in range(200):
        x = x - alpha * (H @ x)
    print(alpha, f"{np.linalg.norm(x):.2e}")
```

```text
[0.31623 0.12403 0.04872 0.01932 0.0077  0.00308 0.00123 0.00049]
ratios of consecutive errors: [0.392 0.393 0.397 0.399 0.399 0.4   0.4  ]
Rayleigh estimate after 8 steps: 5.000491
0.19 7.06e-10
0.21 1.90e+08
```

The errors of the power method shrink by a factor close to $0.4 = |\lambda_2/\lambda_1| = 2/5$ at each step (the ratios of consecutive errors approach $0.4$), and the Rayleigh estimate approaches $5$. For the quadratic, $\lambda_{\max} = 10$ gives the limit $2/\lambda_{\max} = 0.2$. A step of $0.19$ is below the limit and the iterates reach almost zero; a step of $0.21$ is above it, and the iterates grow without bound, because along the eigenvector for $10$ the error is multiplied by $1 - 10\alpha$, which has absolute value above $1$ when $\alpha > 0.2$.
