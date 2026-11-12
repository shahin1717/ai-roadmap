# Solutions: Vector Spaces and Subspaces

Worked answers to the [problems](problems.md). The code outputs were produced by running the code. Back to the [topic](../README.md).

## 1. Subspace or not?

- (a) Yes. It is the line $y = 3x$ through the origin. Two points $(a, 3a)$ and $(b, 3b)$ add to $(a + b, 3(a + b))$, and $c(a, 3a) = (ca, 3ca)$, both on the line.
- (b) No. It does not contain $\mathbf{0}$ (since $0 + 0 \ne 1$). Also $(1, 0) + (0, 1) = (1, 1)$ has $x + y = 2$.
- (c) No. It is closed under addition, but $-1 \cdot (1, 1) = (-1, -1)$ leaves the set.
- (d) Yes. If $x = y$ for two vectors, then it holds for their sum and for any multiple. It is a plane through the origin.
- (e) No. It is the two axes together: $(1, 0)$ and $(0, 1)$ are in it, but their sum $(1, 1)$ has $xy = 1$.

## 2. A column space

(a) Every combination is $x_1(1,0,1) + x_2(0,1,1) = (x_1, x_2, x_1 + x_2)$, so $C(A)$ is the plane $b_3 = b_1 + b_2$.

(b) For $(2, 3, 5)$: $5 = 2 + 3$, so yes (with $x_1 = 2$, $x_2 = 3$). For $(1, 1, 3)$: $3 \ne 2$, so no.

(c) Subtract row 1 and then row 2 from row 3: the last row of $[A \mid \mathbf{b}]$ becomes $(0, 0 \mid b_3 - b_1 - b_2)$. In the unsolvable case this is $0 = b_3 - b_1 - b_2 \ne 0$, a contradiction. For $\mathbf{b} = (1, 1, 3)$ it reads $0 = 1$.

```python
import numpy as np

A = np.array([[1, 0], [0, 1], [1, 1]])
for b in ([2, 3, 5], [1, 1, 3]):
    print(b, np.linalg.matrix_rank(np.column_stack([A, b])) == np.linalg.matrix_rank(A))
```

```text
[2, 3, 5] True
[1, 1, 3] False
```

## 3. By inspection

For $`\begin{bmatrix} 1 & 2 \\ 3 & 6 \end{bmatrix}`$ the second row is $3\times$ the first, so $A\mathbf{x} = \mathbf{0}$ is the single equation $x_1 + 2x_2 = 0$. Its solutions are the multiples of $(-2, 1)$: a line through the origin.

For $`\begin{bmatrix} 1 & 2 \\ 3 & 7 \end{bmatrix}`$, row 2 minus $3\times$ row 1 gives $(0, 1)$, so both columns have pivots. $x_2 = 0$ from the second equation, and then $x_1 = 0$ from the first. Only $\mathbf{0}$ solves it.

## 4. Row reduce

Row 2 minus $2\times$ row 1: $(0, 0, 1, 2)$. Row 3 minus $3\times$ row 1: $(0, 0, 1, 2)$. Row 3 minus the new row 2: $(0, 0, 0, 0)$. Row 1 minus row 2 clears the entry above the second pivot:

```math
R = \begin{bmatrix} 1 & 2 & 0 & 1 \\ 0 & 0 & 1 & 2 \\ 0 & 0 & 0 & 0 \end{bmatrix}.
```

The pivot columns are $1$ and $3$, the free columns are $2$ and $4$, and the rank is $r = 2$.

## 5. Special solutions

From $R\mathbf{x} = \mathbf{0}$: $x_1 = -2x_2 - x_4$ and $x_3 = -2x_4$. Taking $(x_2, x_4) = (1, 0)$ gives $\mathbf{s}_1 = (-2, 1, 0, 0)$, and $(0, 1)$ gives $\mathbf{s}_2 = (-1, 0, -2, 1)$. Check $\mathbf{s}_2$ in $A$: row 1 gives $-1 + 0 - 2 + 3 = 0$, row 2 gives $-2 + 0 - 6 + 8 = 0$, row 3 gives $-3 + 0 - 8 + 11 = 0$. The nullspace is $\{x_2\mathbf{s}_1 + x_4\mathbf{s}_2\}$, a plane through the origin in $\mathbb{R}^4$.

```python
A = np.array([[1, 2, 1, 3], [2, 4, 3, 8], [3, 6, 4, 11]])
s1 = np.array([-2, 1, 0, 0])
s2 = np.array([-1, 0, -2, 1])
print(A @ s1, A @ s2)
```

```text
[0 0 0] [0 0 0]
```

## 6. The complete solution

Row reduce $[A \mid \mathbf{b}]$ with $\mathbf{b} = (1, 3, 4)$: row 2 minus $2\times$ row 1 gives $(0,0,1,2 \mid 1)$, row 3 minus $3\times$ row 1 gives $(0,0,1,2 \mid 1)$, row 3 minus row 2 gives $(0,0,0,0 \mid 0)$, and row 1 minus row 2 gives $(1,2,0,1 \mid 0)$. So $[R \mid \mathbf{d}]$ has rows $(1,2,0,1 \mid 0)$, $(0,0,1,2 \mid 1)$, $(0,0,0,0 \mid 0)$. The zero row ends in $0$, so the system is solvable. With the free variables $x_2 = x_4 = 0$ we get $x_1 = 0$ and $x_3 = 1$, so $\mathbf{x}_p = (0, 0, 1, 0)$ and

$$
\mathbf{x} = (0, 0, 1, 0) + x_2\,(-2, 1, 0, 0) + x_4\,(-1, 0, -2, 1).
$$

For $\mathbf{b} = (1, 3, 5)$, the same steps give row 3 minus row 2 as $(0,0,0,0 \mid 1)$, which says $0 = 1$: no solution. In terms of the column space, $(1, 3, 5)$ is not in $C(A)$ (note that row 3 of $A$ is row 1 plus row 2, so $b_3$ must equal $b_1 + b_2 = 4$).

```python
x_p = np.array([0, 0, 1, 0])
x = x_p + 2 * s1 - 5 * s2          # any choice of the two free variables works
print(A @ x)
```

```text
[1 3 4]
```

## 7. Four cases

| Matrix | $m$ | $n$ | $r$ | Case | Solutions |
|---|---|---|---|---|---|
| (a) rows $(1,2,3)$, $(0,1,4)$ | 2 | 3 | 2 | wide, full row rank | always solvable, infinitely many |
| (b) the matrix of problem 2 | 3 | 2 | 2 | tall, full column rank | none or exactly one |
| (c) rows $(1,2)$, $(2,4)$ | 2 | 2 | 1 | neither | none or infinitely many |
| (d) rows $(2,1)$, $(1,1)$ | 2 | 2 | 2 | square, invertible | exactly one, for every $\mathbf{b}$ |

For (c), the second row is twice the first, so there is only one pivot. For (d), the determinant is $2\cdot1 - 1\cdot1 = 1 \ne 0$, so there are two pivots.

```python
for M in ([[1, 2, 3], [0, 1, 4]], [[1, 0], [0, 1], [1, 1]], [[1, 2], [2, 4]], [[2, 1], [1, 1]]):
    M = np.array(M)
    print(M.shape, np.linalg.matrix_rank(M))
```

```text
(2, 3) 2
(3, 2) 2
(2, 2) 1
(2, 2) 2
```

## 8. A line in space

Let $B$ have rows $(1,1,1)$ and $(1,2,3)$. Row 2 minus row 1 gives $(0,1,2)$, and then row 1 minus that gives $(1,0,-1)$. So $R$ has rows $(1,0,-1)$ and $(0,1,2)$, with pivot columns $1$ and $2$ and one free column (column $3$). Set $x_3 = 1$: then $x_1 = 1$ and $x_2 = -2$. The special solution is $(1, -2, 1)$, and the nullspace is all multiples of it: a line through the origin in $\mathbb{R}^3$. Geometrically, each equation is a plane through the origin, and the two planes meet in exactly this line.

```python
B = np.array([[1, 1, 1], [1, 2, 3]])
print(B @ np.array([1, -2, 1]))
```

```text
[0 0]
```

## 9. Shifted solution sets

Check with $A$: $A\,(1, 2) = (1 + 4,\ 2 + 8) = (5, 10)$ and $A\,(3, 1) = (3 + 2,\ 6 + 4) = (5, 10)$. The difference is $(3, 1) - (1, 2) = (2, -1)$, and $A\,(2, -1) = (2 - 2,\ 4 - 4) = (0, 0)$, so it lies in the nullspace (it is a multiple of $(-2, 1)$). The complete solution with $\mathbf{x}_p = (5, 0)$ is $(5, 0) + t(-2, 1)$. With $\mathbf{x}_p = (1, 2)$ it is $(1, 2) + t(-2, 1)$. Both describe the same line, because $(1, 2) = (5, 0) + 2\,(-2, 1)$ is itself on the first line. The two descriptions differ only in where along the line they start, so any particular solution can serve as $\mathbf{x}_p$.

```python
A = np.array([[1, 2], [2, 4]])
print(A @ [1, 2], A @ [3, 1], A @ [5, 0])
```

```text
[ 5 10] [ 5 10] [ 5 10]
```

## 10. Code it

```python
import sympy as sp


def rref(A, tol=1e-12):
    """Reduce A to reduced row echelon form. Returns R and the list of pivot columns."""
    R = np.array(A, dtype=float)
    m, n = R.shape
    pivots, row = [], 0
    for col in range(n):
        if row == m:
            break
        below = [i for i in range(row, m) if abs(R[i, col]) > tol]
        if not below:
            continue                                   # a free column
        R[[row, below[0]]] = R[[below[0], row]]
        R[row] = R[row] / R[row, col]                  # pivot becomes 1
        for i in range(m):
            if i != row:
                R[i] = R[i] - R[i, col] * R[row]       # clear the column
        pivots.append(col)
        row += 1
    return R, pivots


def special_solutions(R, pivots):
    """One special solution per free column, as the columns of a matrix."""
    n = R.shape[1]
    free = [j for j in range(n) if j not in pivots]
    N = np.zeros((n, len(free)))
    for k, j in enumerate(free):
        N[j, k] = 1
        for row, pivot_col in enumerate(pivots):
            N[pivot_col, k] = -R[row, j]
    return N


# (a) the matrix of problem 4, compared with SymPy
A4 = np.array([[1, 2, 1, 3], [2, 4, 3, 8], [3, 6, 4, 11]])
R, pivots = rref(A4)
sym_R, sym_pivots = sp.Matrix(A4).rref()
print(np.allclose(R, np.array(sym_R.tolist(), dtype=float)), pivots == list(sym_pivots))

# (b) a random 5 x 6 matrix of rank 2
rng = np.random.default_rng(0)
A = rng.normal(size=(5, 2)) @ rng.normal(size=(2, 6))
R, pivots = rref(A)
N = special_solutions(R, pivots)
print(np.linalg.matrix_rank(A), N.shape[1], np.allclose(A @ N, 0))
```

```text
True True
2 4 True
```

Both tests pass: our row reduction matches SymPy's on the integer example, and for the random rank-2 matrix the rank is 2, there are $6 - 2 = 4$ special solutions, and each one is sent to zero by $A$.
