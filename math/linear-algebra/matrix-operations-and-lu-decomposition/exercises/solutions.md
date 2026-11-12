# Solutions: Matrix Operations and LU Decomposition

Worked answers to the [problems](problems.md). The code outputs were produced by running the code. Back to the [topic](../README.md).

## 1. Two ways to multiply

(a) Each entry is a row of $A$ dotted with a column of $B$:
```math
AB = \begin{bmatrix} 2\cdot1 + 0\cdot2 & 2\cdot4 + 0\cdot0 \\ 1\cdot1 + 3\cdot2 & 1\cdot4 + 3\cdot0 \end{bmatrix}
= \begin{bmatrix} 2 & 8 \\ 7 & 4 \end{bmatrix}.
```
(b) $A\,(1, 2) = 1\cdot(2, 1) + 2\cdot(0, 3) = (2, 7)$, the first column of $AB$.

(c)
```math
BA = \begin{bmatrix} 1\cdot2 + 4\cdot1 & 1\cdot0 + 4\cdot3 \\ 2\cdot2 + 0\cdot1 & 2\cdot0 + 0\cdot3 \end{bmatrix}
= \begin{bmatrix} 6 & 12 \\ 4 & 0 \end{bmatrix}.
```
The two products are different.

## 2. True or false?

- (a) True. Both groupings mean "apply $C$, then $B$, then $A$".
- (b) False. Example: $`A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}`$, $`B = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}`$ give $`AB = \begin{bmatrix} 2 & 1 \\ 1 & 1 \end{bmatrix}`$ and $`BA = \begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix}`$.
- (c) True. This is the distributive law.
- (d) False in general. Expanding gives $A^2 + AB + BA + B^2$, and $AB + BA$ equals $2AB$ only when $AB = BA$. The matrices of (b) are a counterexample.
- (e) False. Take $A$ with a single $1$ in the top-left corner and $B$ with a single $1$ in the bottom-right corner. Both are nonzero, yet $AB$ is the zero matrix.

## 3. Powers, and two layers in one

(a)
```math
A^2 = \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix},\quad A^3 = \begin{bmatrix} 1 & 3 \\ 0 & 1 \end{bmatrix},\quad\text{so}\quad A^n = \begin{bmatrix} 1 & n \\ 0 & 1 \end{bmatrix}.
```
Each extra factor of $A$ adds one to the top-right entry.

(b) $W_1\mathbf{x} = (1 + 6,\ 2 + 3) = (7, 5)$ and $W_2(7, 5) = (12, 14, 15)$. The product $W_2W_1$ is $3 \times 3$ (a $3\times2$ times a $2\times3$) with rows $(1,1,3)$, $(2,0,4)$, $(0,3,3)$, and applying it to $\mathbf{x}$ gives $(1 + 2 + 9,\ 2 + 12,\ 6 + 9) = (12, 14, 15)$, the same answer. Two linear steps collapse into one matrix.

```python
import numpy as np

W1 = np.array([[1, 0, 2], [0, 1, 1]])
W2 = np.array([[1, 1], [2, 0], [0, 3]])
x = np.array([1, 2, 3])
print(W2 @ (W1 @ x))
print(W2 @ W1)
print((W2 @ W1) @ x)
```

```text
[12 14 15]
[[1 1 3]
 [2 0 4]
 [0 3 3]]
[12 14 15]
```

## 4. A 2 by 2 inverse

(a) $\det A = 4\cdot6 - 7\cdot2 = 10$.

(b) Swap the diagonal entries, negate the off-diagonal ones, and divide by the determinant:
```math
A^{-1} = \frac{1}{10}\begin{bmatrix} 6 & -7 \\ -2 & 4 \end{bmatrix} = \begin{bmatrix} 0.6 & -0.7 \\ -0.2 & 0.4 \end{bmatrix}.
```
Check: $A A^{-1}$ has first row $(4\cdot0.6 + 7\cdot(-0.2),\ 4\cdot(-0.7) + 7\cdot0.4) = (1, 0)$ and second row $(2\cdot0.6 + 6\cdot(-0.2),\ 2\cdot(-0.7) + 6\cdot0.4) = (0, 1)$.

(c) $\mathbf{x} = A^{-1}(1, 0)$ is the first column of $A^{-1}$: $\mathbf{x} = (0.6, -0.2)$. Check: $4\cdot0.6 + 7\cdot(-0.2) = 1$ and $2\cdot0.6 + 6\cdot(-0.2) = 0$.

(d) The determinant is $1\cdot4 - 2\cdot2 = 0$. Equivalently, elimination gives $R_2 - 2R_1 = (0, 0)$, so there is no second pivot.

## 5. Gauss-Jordan

Start with $[A \mid I]$. The matrix is already upper triangular with pivots $1$, so we only clear above the pivots, working from the bottom:

```math
\left[\begin{array}{ccc|ccc} 1 & 2 & 3 & 1 & 0 & 0 \\ 0 & 1 & 4 & 0 & 1 & 0 \\ 0 & 0 & 1 & 0 & 0 & 1 \end{array}\right]
\xrightarrow{\substack{R_2 - 4R_3 \\ R_1 - 3R_3}}
\left[\begin{array}{ccc|ccc} 1 & 2 & 0 & 1 & 0 & -3 \\ 0 & 1 & 0 & 0 & 1 & -4 \\ 0 & 0 & 1 & 0 & 0 & 1 \end{array}\right]
\xrightarrow{R_1 - 2R_2}
\left[\begin{array}{ccc|ccc} 1 & 0 & 0 & 1 & -2 & 5 \\ 0 & 1 & 0 & 0 & 1 & -4 \\ 0 & 0 & 1 & 0 & 0 & 1 \end{array}\right]
```

So $A^{-1}$ has rows $(1, -2, 5)$, $(0, 1, -4)$, $(0, 0, 1)$. Check the first row of $AA^{-1}$: $(1\cdot1,\ 1\cdot(-2) + 2\cdot1,\ 1\cdot5 + 2\cdot(-4) + 3\cdot1) = (1, 0, 0)$.

```python
A = np.array([[1, 2, 3], [0, 1, 4], [0, 0, 1]])
print(np.linalg.inv(A))
print(A @ np.linalg.inv(A))
```

```text
[[ 1. -2.  5.]
 [ 0.  1. -4.]
 [ 0.  0.  1.]]
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

## 6. Determinants

- (a) $3\cdot4 - 1\cdot2 = 10$.
- (b) The pivots are $2$, $1$, $-5$ with no exchanges, so $\det = 2\cdot1\cdot(-5) = -10$.
- (c) After swapping rows 1 and 2, the pivots are $1$, $2$, $\tfrac32$. One exchange flips the sign: $\det = -(1\cdot2\cdot\tfrac32) = -3$.
- (d) $\det A = 1\cdot4 - 2\cdot3 = -2$ and $\det B = 0\cdot2 - 1\cdot1 = -1$, so the product is $2$. Also $AB$ has rows $(2, 5)$ and $(4, 11)$, with determinant $2\cdot11 - 5\cdot4 = 2$. They agree.

```python
print(round(np.linalg.det(np.array([[3, 1], [2, 4]])), 6))
print(round(np.linalg.det(np.array([[2, 1, -1], [4, 3, 1], [-2, 2, 5]])), 6))
print(round(np.linalg.det(np.array([[0, 2, 1], [1, 1, 1], [2, 1, 3]])), 6))
print(round(np.linalg.det(np.array([[2, 5], [4, 11]])), 6))
```

```text
10.0
-10.0
-3.0
2.0
```

## 7. LU by hand

The first pivot is $1$.

- Multiplier $\ell_{21} = 3/1 = 3$: row 2 minus $3\times$ row 1 gives $(0, 2, -2)$.
- Multiplier $\ell_{31} = 0/1 = 0$: row 3 is unchanged, $(0, 4, 1)$.
- The second pivot is $2$. Multiplier $\ell_{32} = 4/2 = 2$: row 3 minus $2\times$ row 2 gives $(0, 0, 5)$.

```math
L = \begin{bmatrix} 1 & 0 & 0 \\ 3 & 1 & 0 \\ 0 & 2 & 1 \end{bmatrix},\qquad
U = \begin{bmatrix} 1 & 2 & 1 \\ 0 & 2 & -2 \\ 0 & 0 & 5 \end{bmatrix}.
```

Check: row 2 of $LU$ is $3\,(1,2,1) + (0,2,-2) = (3, 8, 1)$, and row 3 is $0 + 2\,(0,2,-2) + (0,0,5) = (0, 4, 1)$. Both match $A$.

```python
L = np.array([[1, 0, 0], [3, 1, 0], [0, 2, 1]])
U = np.array([[1, 2, 1], [0, 2, -2], [0, 0, 5]])
print(L @ U)
```

```text
[[1 2 1]
 [3 8 1]
 [0 4 1]]
```

## 8. Many right-hand sides

For $\mathbf{b} = (2, 12, 2)$: forward substitution $L\mathbf{c} = \mathbf{b}$ gives $c_1 = 2$, $c_2 = 12 - 3\cdot2 = 6$, $c_3 = 2 - 0\cdot2 - 2\cdot6 = -10$. Back substitution $U\mathbf{x} = \mathbf{c}$ gives $5z = -10$, so $z = -2$; then $2y - 2z = 6$, so $y = 1$; then $x + 2y + z = 2$, so $x = 2$. The solution is $(2, 1, -2)$.

For $\mathbf{b} = (1, 0, 0)$: $c_1 = 1$, $c_2 = -3$, $c_3 = 0 - 0 - 2\cdot(-3) = 6$. Back: $z = \tfrac65$, then $2y = -3 + 2z = -\tfrac35$ so $y = -\tfrac3{10}$, then $x = 1 - 2y - z = 1 + \tfrac35 - \tfrac65 = \tfrac25$. The solution is $(\tfrac25, -\tfrac3{10}, \tfrac65)$.

Cost for $n = 1000$: the factorization takes about $n^3/3 \approx 3.3\times10^8$ multiplications, and each extra solve takes about $n^2 = 10^6$. That is about 300 times cheaper per extra right-hand side.

```python
A = L @ U
print(np.linalg.solve(A, [2, 12, 2]))
print(np.linalg.solve(A, [1, 0, 0]))
```

```text
[ 2.  1. -2.]
[ 0.4 -0.3  1.2]
```

## 9. Transposes and permutations

(a) $`AB = \begin{bmatrix} 5 & 4 \\ 1 & 2 \end{bmatrix}`$, so $`(AB)^\top = \begin{bmatrix} 5 & 1 \\ 4 & 2 \end{bmatrix}`$. Also $`B^\top = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}`$ and $`A^\top = \begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix}`$, and $`B^\top A^\top = \begin{bmatrix} 5 & 1 \\ 4 & 2 \end{bmatrix}`$. They agree.

(b) $`A^\top A = \begin{bmatrix} 35 & 44 \\ 44 & 56 \end{bmatrix}`$. It is symmetric because $(A^\top A)^\top = A^\top (A^\top)^\top = A^\top A$ (the entry in row $i$, column $j$ is the dot product of columns $i$ and $j$ of $A$, which equals the entry in row $j$, column $i$).

(c) $P\mathbf{v} = (8, 9, 7)$: each entry moves up one place, and the first goes to the end. And $P^\top P = I$, so $P^{-1} = P^\top$.

```python
A = np.array([[1, 2], [0, 1]])
B = np.array([[3, 0], [1, 2]])
print(np.array_equal((A @ B).T, B.T @ A.T))
G = np.array([[1, 2], [3, 4], [5, 6]])
print(G.T @ G)
P = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
print(P @ np.array([7, 8, 9]))
print(P.T @ P)
```

```text
True
[[35 44]
 [44 56]]
[8 9 7]
[[1 0 0]
 [0 1 0]
 [0 0 1]]
```

## 10. Write an LU routine

```python
import scipy.linalg as sla


def lu_decompose(A):
    """Return P, L, U with A = P @ L @ U, using partial pivoting."""
    U = np.array(A, dtype=float)
    n = len(U)
    L = np.eye(n)
    perm = np.arange(n)
    for k in range(n):
        p = k + np.argmax(np.abs(U[k:, k]))          # largest entry in column k
        if abs(U[p, k]) < 1e-12:
            raise ValueError(f"no pivot in column {k}")
        if p != k:
            U[[k, p]] = U[[p, k]]
            perm[[k, p]] = perm[[p, k]]
            L[[k, p], :k] = L[[p, k], :k]            # swap the multipliers stored so far
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]              # the multiplier goes into L
            U[i, k:] -= L[i, k] * U[k, k:]
    P = np.eye(n)[perm].T
    return P, L, U


rng = np.random.default_rng(0)
A = rng.normal(size=(6, 6))
P, L, U = lu_decompose(A)
P2, L2, U2 = sla.lu(A)
print(np.allclose(P @ L @ U, A), np.allclose(P, P2), np.allclose(L, L2), np.allclose(U, U2))

# (a) determinant from U and the sign of P
sign = np.linalg.det(P)                              # +1 or -1
print(np.isclose(sign * np.prod(np.diag(U)), np.linalg.det(A)))

# (b) one factorization, 20 right-hand sides
rhs = rng.normal(size=(6, 20))
lu_piv = sla.lu_factor(A)
X = np.column_stack([sla.lu_solve(lu_piv, rhs[:, j]) for j in range(20)])
print(np.allclose(X, np.linalg.solve(A, rhs)))
```

```text
True True True True
True
True
```

The factors match SciPy's. The determinant is the product of the diagonal of $U$ (the pivots) times the sign of $P$, which accounts for the row exchanges, matching the rule of the lesson. And one factorization serves all twenty right-hand sides.
