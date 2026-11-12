# Solutions: Independence, Basis, and Dimension

Worked answers to the [problems](problems.md). The code outputs were produced by running the code. Back to the [topic](../README.md).

## 1. Independent or dependent?

- (a) Independent. The matrix with these columns has rows $(1,3)$ and $(2,4)$. Row 2 minus $2\times$ row 1 gives $(0,-2)$, so there are two pivots. (Equivalently, $\det = 1\cdot4 - 3\cdot2 = -2 \ne 0$.)
- (b) Dependent: $2\,(1,2) - 1\,(2,4) = (0,0)$.
- (c) Dependent: $1\,(1,0,0) + 1\,(0,1,0) - 1\,(1,1,0) = (0,0,0)$.
- (d) Dependent, as any three vectors in $\mathbb{R}^2$ are: $5\,(1,0) + 7\,(0,1) - 1\,(5,7) = (0,0)$.

## 2. The matrix test

The matrix has columns $(1,2,3)$, $(4,5,6)$, $(7,8,9)$. Row 2 minus $2\times$ row 1 gives $(0,-3,-6)$, and row 3 minus $3\times$ row 1 gives $(0,-6,-12)$. Row 3 minus $2\times$ the new row 2 gives zeros. There are two pivots, so the rank is $2$, not $3$, and the third column is free. The special solution is $(1, -2, 1)$: it says $1\,(1,2,3) - 2\,(4,5,6) + 1\,(7,8,9) = \mathbf{0}$, so the third column equals twice the second minus the first. The columns are dependent, and the nullspace is all multiples of $(1,-2,1)$.

```python
import numpy as np

A = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
print(np.linalg.matrix_rank(A))
print(A @ np.array([1, -2, 1]))
```

```text
2
[0 0 0]
```

## 3. Spans

(a) The second vector is twice the first, so the span is the line through the origin in the direction $(1,2,1)$.

(b) A combination is $x_1(1,0,1) + x_2(0,1,1) = (x_1, x_2, x_1 + x_2)$, so the span is the plane $z = x + y$. For $(3,2,5)$: $3 + 2 = 5$, so yes.

(c) The span is all of $\mathbb{R}^3$, since the first three vectors already span it. Only three of the four are needed; $(7,7,7)$ is dependent on the others: it equals $7\,(1,0,0) + 7\,(0,1,0) + 7\,(0,0,1)$.

## 4. Coordinates

Solve $c_1(1,2) + c_2(2,3) = (4,7)$: $c_1 + 2c_2 = 4$ and $2c_1 + 3c_2 = 7$. From the first, $c_1 = 4 - 2c_2$, so $8 - 4c_2 + 3c_2 = 7$, giving $c_2 = 1$ and $c_1 = 2$. The coordinates are $(2, 1)$. Check: $2\,(1,2) + (2,3) = (4,7)$.

The vectors $(1,2)$ and $(2,4)$ are dependent ($(2,4)$ is twice $(1,2)$), and a basis must be independent. They span only a line, not the plane.

```python
B = np.array([[1, 2], [2, 3]])
print(np.linalg.solve(B, [4, 7]))
```

```text
[2. 1.]
```

## 5. A basis for a plane

The plane is $N(A)$ for $A = [\,1\ -1\ 2\,]$. Column 1 is the pivot column, and columns 2 and 3 are free. Setting $(x_2, x_3) = (1, 0)$ gives $x_1 = 1$, so $(1,1,0)$. Setting $(0, 1)$ gives $x_1 = -2$, so $(-2,0,1)$. A basis is $(1,1,0)$ and $(-2,0,1)$, and the dimension is $2$ (the number of free columns, $3 - 1$). Check: $1 - 1 + 0 = 0$ and $-2 - 0 + 2 = 0$.

```python
import sympy as sp

print([list(v) for v in sp.Matrix([[1, -1, 2]]).nullspace()])
```

```text
[[1, 1, 0], [-2, 0, 1]]
```

## 6. A basis from vectors

With the vectors as columns, the matrix is $`\begin{bmatrix} 1&2&3&1 \\ 1&2&4&0 \\ 0&0&1&-1 \end{bmatrix}`$ (its columns are $\mathbf{v}_1$ to $\mathbf{v}_4$). Row 2 minus row 1 gives $(0,0,1,-1)$; then row 3 minus that gives zeros; then row 1 minus $3\times$ row 2 gives $(1,2,0,4)$:

```math
R = \begin{bmatrix} 1 & 2 & 0 & 4 \\ 0 & 0 & 1 & -1 \\ 0 & 0 & 0 & 0 \end{bmatrix}.
```

The pivot columns are $1$ and $3$, so $\mathbf{v}_1$ and $\mathbf{v}_3$ are a basis and the dimension is $2$. The free columns of $R$ give the recipes: $\mathbf{v}_2 = 2\mathbf{v}_1$, and $\mathbf{v}_4 = 4\mathbf{v}_1 - \mathbf{v}_3$. Check: $4\,(1,1,0) - (3,4,1) = (1,0,-1)$.

```python
V = np.array([[1, 2, 3, 1], [1, 2, 4, 0], [0, 0, 1, -1]])
R, pivots = sp.Matrix(V).rref()
print(R)
print(pivots)
print(4 * V[:, 0] - V[:, 2])
```

```text
Matrix([[1, 2, 0, 4], [0, 0, 1, -1], [0, 0, 0, 0]])
(0, 2)
[ 1  0 -1]
```

## 7. All four at once

Rows $1$ and $2$ of $A$ are already in echelon form, and row 3 equals row 1 plus row 2, so one step makes it zero:

```math
R = \begin{bmatrix} 1 & 2 & 0 & 3 \\ 0 & 0 & 1 & 4 \\ 0 & 0 & 0 & 0 \end{bmatrix}.
```

The rank is $r = 2$, with $m = 3$ and $n = 4$. Pivot columns: $1$ and $3$. Free columns: $2$ and $4$.

- Column space (dimension $2$, in $\mathbb{R}^3$): the pivot columns of $A$, $(1,0,1)$ and $(0,1,1)$.
- Row space (dimension $2$, in $\mathbb{R}^4$): the nonzero rows of $R$, $(1,2,0,3)$ and $(0,0,1,4)$.
- Nullspace (dimension $4 - 2 = 2$, in $\mathbb{R}^4$): the special solutions $(-2,1,0,0)$ and $(-3,0,-4,1)$.
- Left nullspace (dimension $3 - 2 = 1$, in $\mathbb{R}^3$): row 3 is row 1 plus row 2, so $\mathbf{y} = (1,1,-1)$ gives $\mathbf{y}^\top A = \mathbf{0}$.

Dimensions: $2 + 2 = 4 = n$ and $2 + 1 = 3 = m$. Perpendicularity: the row $(1,2,0,3)$ dotted with the special solution $(-3,0,-4,1)$ is $-3 + 0 + 0 + 3 = 0$.

```python
A = np.array([[1, 2, 0, 3], [0, 0, 1, 4], [1, 2, 1, 7]])
print(np.array([1, 1, -1]) @ A)
print(A @ np.array([-2, 1, 0, 0]), A @ np.array([-3, 0, -4, 1]))
print(np.array([1, 2, 0, 3]) @ np.array([-3, 0, -4, 1]))
```

```text
[0 0 0 0]
[0 0 0] [0 0 0]
0
```

## 8. Counting dimensions

| | Shape, rank | Column space | Row space | Nullspace | Left nullspace | Solvable for every $\mathbf{b}$? Unique? |
|---|---|---|---|---|---|---|
| (a) | $5 \times 8$, $r = 3$ | 3 | 3 | $8 - 3 = 5$ | $5 - 3 = 2$ | no (some $\mathbf{b}$ lie outside the column space); when a solution exists it is not unique, because there are 5 free columns |
| (b) | $6 \times 6$, $r = 6$ | 6 | 6 | 0 | 0 | yes, and unique (invertible) |
| (c) | $4 \times 7$, $r = 4$ | 4 | 4 | $7 - 4 = 3$ | 0 | yes, but never unique (3 free columns) |

(d) The rank cannot exceed the number of rows, since there are at most $5$ pivots in $5$ rows. So $r \le 5$, and rank $6$ is impossible.

## 9. Rank of a product

(a) At most $2$. The columns of $AB$ are combinations of the columns of $A$, so $C(AB)$ sits inside $C(A)$, which has dimension $2$. In general, $\operatorname{rank}(AB) \le \min(\operatorname{rank} A, \operatorname{rank} B)$, here $\min(2, 3) = 2$.

(b) At most $2$, since $W_1$ has only $2$ rows (rank at most $2$). The $6 \times 6$ matrix $W_2W_1$ therefore has rank at most $2$: every output lies in a $2$-dimensional subspace of $\mathbb{R}^6$, however many inputs we try.

```python
rng = np.random.default_rng(0)
A = rng.normal(size=(5, 2)) @ rng.normal(size=(2, 3))      # 5 x 3, rank 2
B = rng.normal(size=(3, 4))                                # 3 x 4, rank 3 (almost surely)
print(np.linalg.matrix_rank(A), np.linalg.matrix_rank(B), np.linalg.matrix_rank(A @ B))

W1, W2 = rng.normal(size=(2, 6)), rng.normal(size=(6, 2))
print((W2 @ W1).shape, np.linalg.matrix_rank(W2 @ W1))
```

```text
2 3 2
(6, 6) 2
```

## 10. Write the tools

```python
import scipy.linalg as sla


def is_independent(vectors):
    """Return (True, None) if the vectors are independent, else (False, coefficients)
    of a nontrivial combination that equals zero."""
    A = np.column_stack(vectors).astype(float)
    if np.linalg.matrix_rank(A) == A.shape[1]:
        return True, None
    c = sla.null_space(A)[:, 0]
    return False, c / c[np.argmax(np.abs(c) > 1e-9)]      # scale so the first nonzero entry is 1


def basis_for_span(vectors):
    """Return a matrix whose columns are a basis: the pivot columns of the original matrix."""
    A = np.column_stack(vectors)
    _, pivots = sp.Matrix(A).rref()
    return A[:, list(pivots)]


print(is_independent([(1, 2), (3, 4)]))
print(is_independent([(1, 0, 0), (0, 1, 0), (1, 1, 0)]))
print(is_independent([(1, 2, 3), (4, 5, 6), (7, 8, 9)]))
print(basis_for_span([(1, 1, 0), (2, 2, 0), (3, 4, 1), (1, 0, -1)]))

rng = np.random.default_rng(1)
ok = 0
for _ in range(100):
    m, n = rng.integers(2, 9, size=2)
    k = rng.integers(1, min(m, n) + 1)
    M = rng.normal(size=(m, k)) @ rng.normal(size=(k, n))
    ok += np.linalg.matrix_rank(M) + sla.null_space(M).shape[1] == n
print(ok, "of 100 matrices satisfy r + dim N(A) = n")
```

```text
(True, None)
(False, array([ 1.,  1., -1.]))
(False, array([ 1., -2.,  1.]))
[[1 3]
 [1 4]
 [0 1]]
100 of 100 matrices satisfy r + dim N(A) = n
```

The checks agree with the hand answers: the first list is independent, the second and third are dependent with the relations found above (up to scaling), the basis is the pair $\mathbf{v}_1$, $\mathbf{v}_3$, and rank-nullity held for all one hundred random matrices.
