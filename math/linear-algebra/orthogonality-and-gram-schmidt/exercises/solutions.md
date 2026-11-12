# Solutions: Orthogonality and Gram-Schmidt

Worked answers to the [problems](problems.md). The code outputs were produced by running the code. Orthogonality checks on a computer hold only up to rounding error, so the code compares with `np.allclose`. Back to the [topic](../README.md).

## 1. Orthogonal or not?

(a) $(1,2,2)\cdot(2,1,-2) = 2 + 2 - 4 = 0$: orthogonal.

(b) $(1,1)\cdot(1,-2) = 1 - 2 = -1 \ne 0$: not orthogonal.

(c) For the pair in (a): $\mathbf{v} + \mathbf{w} = (3,3,0)$ has squared length $18$, and $\|\mathbf{v}\|^2 + \|\mathbf{w}\|^2 = 9 + 9 = 18$. They agree.

```python
import numpy as np

v = np.array([1, 2, 2]); w = np.array([2, 1, -2])
print(v @ w, (v + w) @ (v + w), v @ v + w @ w)
print(np.array([1, 1]) @ np.array([1, -2]))
```

```text
0 18 18
-1
```

## 2. A complement

A vector $(x,y,z)$ is orthogonal to $(1,2,2)$ exactly when $x + 2y + 2z = 0$, so $V^\perp$ is that plane. It is the nullspace of the $1 \times 3$ matrix $[1\ 2\ 2]$: column 1 is the pivot, and columns 2 and 3 are free. The special solutions are $(-2,1,0)$ and $(-2,0,1)$. A basis for $V^\perp$ is these two vectors, and its dimension is $2$. Check: $-2 + 2 + 0 = 0$ and $-2 + 0 + 2 = 0$. The dimensions add up: $\dim V + \dim V^\perp = 1 + 2 = 3$.

## 3. Four subspaces, two orthogonal pairs

Row 3 is row 1 plus row 2, so one step gives a zero row, and the reduced form has rows $(1,2,0)$, $(0,0,1)$, $(0,0,0)$. The rank is $r = 2$, with pivot columns $1$ and $3$ and free column $2$.

- Column space (dimension $2$, in $\mathbb{R}^3$): the pivot columns of $A$, $(1,0,1)$ and $(0,1,1)$.
- Row space (dimension $2$, in $\mathbb{R}^3$): $(1,2,0)$ and $(0,0,1)$.
- Nullspace (dimension $1$, in $\mathbb{R}^3$): the special solution $(-2,1,0)$.
- Left nullspace (dimension $1$, in $\mathbb{R}^3$): row 1 plus row 2 minus row 3 is zero, so $\mathbf{y} = (1,1,-1)$.

Row space and nullspace: $(1,2,0)\cdot(-2,1,0) = 0$ and $(0,0,1)\cdot(-2,1,0) = 0$. Their dimensions add to $2 + 1 = 3 = n$. Column space and left nullspace: $(1,0,1)\cdot(1,1,-1) = 0$ and $(0,1,1)\cdot(1,1,-1) = 0$, with $2 + 1 = 3 = m$.

```python
A = np.array([[1, 2, 0], [0, 0, 1], [1, 2, 1]])
n_vec = np.array([-2, 1, 0]); y = np.array([1, 1, -1])
print(A @ n_vec)
print(y @ A)
print([np.allclose(row @ n_vec, 0) for row in A])
print(np.linalg.matrix_rank(A))
```

```text
[0 0 0]
[0 0 0]
[True, True, True]
2
```

## 4. Splitting a vector

Write $\mathbf{x}_n = t\,(1,-2,1)$. We need $\mathbf{x} - \mathbf{x}_n$ orthogonal to $(1,-2,1)$: $(1,-1,2)\cdot(1,-2,1) = 1 + 2 + 2 = 5$ and $(1,-2,1)\cdot(1,-2,1) = 6$, so $t = 5/6$. Then $\mathbf{x}_n = (5/6, -5/3, 5/6)$ and $\mathbf{x}_r = \mathbf{x} - \mathbf{x}_n = (1/6, 2/3, 7/6)$. Check that $\mathbf{x}_r$ is in the row space: $a(1,1,1) + b(1,2,3)$ with $a = -1/3$ and $b = 1/2$ gives $(1/6, 2/3, 7/6)$. The two parts are orthogonal: $\tfrac16\cdot\tfrac56 - \tfrac23\cdot\tfrac53 + \tfrac76\cdot\tfrac56 = \tfrac{5}{36} - \tfrac{40}{36} + \tfrac{35}{36} = 0$.

```python
B = np.array([[1, 1, 1], [1, 2, 3]])
x = np.array([1.0, -1.0, 2.0])
n_dir = np.array([1.0, -2.0, 1.0])
x_n = (x @ n_dir) / (n_dir @ n_dir) * n_dir
x_r = x - x_n
print(x_n, x_r)
print(np.allclose(B @ x_n, 0), np.isclose(x_r @ x_n, 0))
print(np.allclose(-1/3 * B[0] + 0.5 * B[1], x_r))
```

```text
[ 0.83333333 -1.66666667  0.83333333] [0.16666667 0.66666667 1.16666667]
True True
True
```

## 5. Orthonormal basis

(a) $\mathbf{q}_1\cdot\mathbf{q}_1 = (1 + 4 + 4)/9 = 1$, $\mathbf{q}_2\cdot\mathbf{q}_2 = (4 + 1 + 4)/9 = 1$, and $\mathbf{q}_1\cdot\mathbf{q}_2 = (2 + 2 - 4)/9 = 0$.

(b) $\mathbf{q}_3\cdot\mathbf{q}_3 = (4 + 4 + 1)/9 = 1$, $\mathbf{q}_1\cdot\mathbf{q}_3 = (2 - 4 + 2)/9 = 0$, and $\mathbf{q}_2\cdot\mathbf{q}_3 = (4 - 2 - 2)/9 = 0$. Three orthonormal vectors in $\mathbb{R}^3$ form an orthonormal basis.

(c) $c_1 = \mathbf{q}_1\cdot\mathbf{v} = (3 + 6 + 0)/3 = 3$, $c_2 = \mathbf{q}_2\cdot\mathbf{v} = (6 + 3 + 0)/3 = 3$, $c_3 = \mathbf{q}_3\cdot\mathbf{v} = (6 - 6 + 0)/3 = 0$. So $\mathbf{v} = 3\mathbf{q}_1 + 3\mathbf{q}_2$. Check: $(1,2,2) + (2,1,-2) = (3,3,0)$.

```python
Q = np.column_stack([np.array([1, 2, 2]) / 3, np.array([2, 1, -2]) / 3, np.array([2, -2, 1]) / 3])
print(np.allclose(Q.T @ Q, np.eye(3)))
print(Q.T @ np.array([3, 3, 0]))
```

```text
True
[3. 3. 0.]
```

## 6. Orthogonal matrices

(a) The columns are $(0.6, -0.8)$ and $(0.8, 0.6)$. Their lengths are $\sqrt{0.36 + 0.64} = 1$, and their dot product is $0.48 - 0.48 = 0$. So $Q$ is orthogonal, and $Q^{-1} = Q^\top$, the matrix with rows $(0.6, -0.8)$ and $(0.8, 0.6)$.

(b) The rotation sends $(3,4)$ to $(0\cdot3 - 1\cdot4,\ 1\cdot3 + 0\cdot4) = (-4, 3)$. Both vectors have length $5$.

(c) The columns $(0.6, 0.8)$ and $(0.8, -0.6)$ have length $1$ and dot product $0.48 - 0.48 = 0$, so the matrix is orthogonal. Its determinant is $0.6\cdot(-0.6) - 0.8\cdot0.8 = -0.36 - 0.64 = -1$, so it is a reflection (it flips orientation).

```python
Qa = np.array([[0.6, 0.8], [-0.8, 0.6]])
Qb = np.array([[0.6, 0.8], [0.8, -0.6]])
print(np.allclose(Qa.T @ Qa, np.eye(2)), np.allclose(Qa.T, np.linalg.inv(Qa)))
print(np.array([[0, -1], [1, 0]]) @ np.array([3, 4]))
print(np.allclose(Qb.T @ Qb, np.eye(2)), round(np.linalg.det(Qb), 6))
```

```text
True True
[-4  3]
True -1.0
```

## 7. A friendly case

$\mathbf{q}_1 = (1,0,0)$, since $\mathbf{a}_1 = (2,0,0)$ has length $2$. Then $\mathbf{q}_1\cdot\mathbf{a}_2 = 1$, and $\mathbf{a}_2 - \mathbf{q}_1 = (0,3,0)$, so $\mathbf{q}_2 = (0,1,0)$. Next, $\mathbf{q}_1\cdot\mathbf{a}_3 = 1$ and $\mathbf{q}_2\cdot\mathbf{a}_3 = 2$, so $\mathbf{a}_3 - \mathbf{q}_1 - 2\mathbf{q}_2 = (0,0,4)$ and $\mathbf{q}_3 = (0,0,1)$. So $Q = I$ and

```math
R = \begin{bmatrix} 2 & 1 & 1 \\ 0 & 3 & 2 \\ 0 & 0 & 4 \end{bmatrix} = A.
```

The matrix $A$ was already upper triangular, which means $\mathbf{a}_j$ only involves the first $j$ coordinate directions. Gram-Schmidt has nothing to fix, and $A = IA$ is its own QR factorization.

```python
def gram_schmidt(A):
    """Classical Gram-Schmidt on the columns of A. Returns Q and R with A = Q R."""
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R


A7 = np.column_stack([(2, 0, 0), (1, 3, 0), (1, 2, 4)])
Q7, R7 = gram_schmidt(A7)
print(Q7)
print(R7)
```

```text
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
[[2. 1. 1.]
 [0. 3. 2.]
 [0. 0. 4.]]
```

## 8. Three vectors

$\mathbf{q}_1 = (1,1,1)/\sqrt3$. Then $\mathbf{q}_1\cdot\mathbf{a}_2 = 6/\sqrt3$, and $\mathbf{a}_2 - \tfrac{6}{\sqrt3}\mathbf{q}_1 = (1,2,3) - 2\,(1,1,1) = (-1, 0, 1)$, of length $\sqrt2$, so $\mathbf{q}_2 = (-1,0,1)/\sqrt2$. Next, $\mathbf{q}_1\cdot\mathbf{a}_3 = 1/\sqrt3$ and $\mathbf{q}_2\cdot\mathbf{a}_3 = -1/\sqrt2$, so
$\mathbf{a}_3 - \tfrac{1}{\sqrt3}\mathbf{q}_1 + \tfrac{1}{\sqrt2}\mathbf{q}_2 = (1,0,0) - (\tfrac13,\tfrac13,\tfrac13) + (-\tfrac12, 0, \tfrac12) = (\tfrac16, -\tfrac13, \tfrac16)$, of length $1/\sqrt6$, so $\mathbf{q}_3 = (1,-2,1)/\sqrt6$.

```math
R = \begin{bmatrix} \sqrt3 & 2\sqrt3 & 1/\sqrt3 \\ 0 & \sqrt2 & -1/\sqrt2 \\ 0 & 0 & 1/\sqrt6 \end{bmatrix}
\approx \begin{bmatrix} 1.732 & 3.464 & 0.577 \\ 0 & 1.414 & -0.707 \\ 0 & 0 & 0.408 \end{bmatrix}.
```

Check orthogonality: $\mathbf{q}_3\cdot\mathbf{q}_1 = (1 - 2 + 1)/\sqrt{18} = 0$ and $\mathbf{q}_3\cdot\mathbf{q}_2 = (-1 + 0 + 1)/\sqrt{12} = 0$.

```python
A8 = np.column_stack([(1, 1, 1), (1, 2, 3), (1, 0, 0)])
Q8, R8 = gram_schmidt(A8)
print(Q8.round(4))
print(R8.round(4))
print(np.allclose(Q8 @ R8, A8), np.allclose(Q8.T @ Q8, np.eye(3)))
```

```text
[[ 0.5774 -0.7071  0.4082]
 [ 0.5774 -0.     -0.8165]
 [ 0.5774  0.7071  0.4082]]
[[ 1.7321  3.4641  0.5774]
 [ 0.      1.4142 -0.7071]
 [ 0.      0.      0.4082]]
True True
```

## 9. Solving with QR

$Q^\top\mathbf{b}$ has entries $\mathbf{q}_1\cdot\mathbf{b} = 0.6\cdot4 + 0.8\cdot11 = 11.2$ and $\mathbf{q}_2\cdot\mathbf{b} = -0.8\cdot4 + 0.6\cdot11 = 3.4$. Back substitution on $R\mathbf{x} = (11.2, 3.4)$: the second row gives $3.4\,x_2 = 3.4$, so $x_2 = 1$; the first row gives $5x_1 + 6.2 = 11.2$, so $x_1 = 1$. The solution is $\mathbf{x} = (1, 1)$, and indeed $A(1,1) = (3 + 1,\ 4 + 7) = (4, 11)$.

```python
A9 = np.array([[3.0, 1.0], [4.0, 7.0]])
Q9, R9 = gram_schmidt(A9)
print(Q9.round(4), R9.round(4))
c = Q9.T @ np.array([4.0, 11.0])
print(c)
print(np.linalg.solve(R9, c))
```

```text
[[ 0.6 -0.8]
 [ 0.8  0.6]] [[5.  6.2]
 [0.  3.4]]
[11.2  3.4]
[1. 1.]
```

## 10. Write and test

```python
import scipy.linalg as sla


def modified_gram_schmidt(A):
    """Modified Gram-Schmidt: remove each component from the remaining columns at once."""
    V = np.array(A, dtype=float)
    m, n = V.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        R[j, j] = np.linalg.norm(V[:, j])
        Q[:, j] = V[:, j] / R[j, j]
        for k in range(j + 1, n):
            R[j, k] = Q[:, j] @ V[:, k]
            V[:, k] -= R[j, k] * Q[:, j]
    return Q, R


def orthogonality_error(Q):
    return np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]))


# (a) the matrix of problem 8
for name, fn in [("classical", gram_schmidt), ("modified", modified_gram_schmidt)]:
    Q, R = fn(A8)
    print(name, np.allclose(Q @ R, A8), np.allclose(Q.T @ Q, np.eye(3)))

# (b) Hilbert matrices
for n in (6, 8):
    H = sla.hilbert(n)
    errors = [orthogonality_error(gram_schmidt(H)[0]),
              orthogonality_error(modified_gram_schmidt(H)[0]),
              orthogonality_error(np.linalg.qr(H)[0])]
    print(n, [f"{e:.1e}" for e in errors])

# (c) random vectors in 50 dimensions
rng = np.random.default_rng(0)
u = rng.normal(size=(5000, 50))
w = rng.normal(size=(5000, 50))
cosines = (u * w).sum(axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1))
print(round(cosines.std(), 4), round(1 / np.sqrt(50), 4))
```

```text
classical True True
modified True True
6 ['2.8e-04', '3.5e-10', '9.6e-16']
8 ['1.4e+00', '3.6e-07', '1.1e-15']
0.1432 0.1414
```

Both versions are correct on the small matrix. On the Hilbert matrices, classical Gram-Schmidt loses orthogonality badly (an error of order $1$ at $n = 8$), modified Gram-Schmidt is much better, and `np.linalg.qr` stays near $10^{-15}$. The exact digits depend on the machine, but the ranking does not. The measured spread of the cosine in 50 dimensions is close to $1/\sqrt{50} \approx 0.141$.
