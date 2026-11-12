# Solutions: Linear Equations and Gaussian Elimination

Worked answers to the [problems](problems.md). The code outputs were produced by running the code. Back to the [topic](../README.md).

## 1. Write it as $A\mathbf{x} = \mathbf{b}$

```math
\begin{bmatrix} 2 & 1 \\ 1 & -3 \end{bmatrix}
\begin{bmatrix} x \\ y \end{bmatrix}
=
\begin{bmatrix} 7 \\ -7 \end{bmatrix}
```

Check: $2\cdot2 + 3 = 7$ and $2 - 3\cdot3 = -7$. The column picture says $2$ copies of the first column plus $3$ copies of the second give $\mathbf{b}$:
$2\,(2, 1) + 3\,(1, -3) = (4 + 3,\ 2 - 9) = (7, -7)$.

## 2. How many solutions?

- (a) The second equation is twice the first, so they describe the same line: infinitely many solutions, such as $(3, 0)$ and $(1, 2)$.
- (b) Doubling the first gives $2x + 2y = 6$, which contradicts $2x + 2y = 7$: no solution (parallel lines).
- (c) Adding the equations gives $2x = 6$, so $x = 3$ and $y = 2$: one solution, $(3, 2)$.

## 3. A 2 by 2 system

The first pivot is $2$. The multiplier is $4/2 = 2$. Row 2 minus $2\times$ row 1: $(4 - 4)x + (-1 - 6)y = 2 - 16$, so $-7y = -14$. The second pivot is $-7$. Back substitution: $y = 2$, then $2x + 6 = 8$, so $x = 1$.

## 4. A 3 by 3 system

Pivot 1 is $1$ (the entry in row 1, column 1).

- Row 2 minus $2\times$ row 1 (multiplier 2): $(0, 1, -1 \mid -2)$.
- Row 3 minus $1\times$ row 1 (multiplier 1): $(0, -2, 1 \mid 3)$.

Pivot 2 is $1$. Row 3 minus $(-2)\times$ row 2 (multiplier $-2$, so add $2\times$ row 2): $(0, 0, -1 \mid -1)$. Pivot 3 is $-1$. The triangular system is

```math
\begin{aligned}
x + y + z &= 2 \\
y - z &= -2 \\
-z &= -1
\end{aligned}
```

Back substitution: $z = 1$, then $y = -2 + z = -1$, then $x = 2 - y - z = 2$. The solution is $(2, -1, 1)$, with pivots $1$, $1$, $-1$.

## 5. A zero pivot

The first equation has no $x$, so position $(1,1)$ holds $0$. Swap the first two equations, which gives rows $(2,1,1 \mid 4)$, $(0,1,2 \mid 3)$, $(1,0,3 \mid 4)$. Pivot 1 is $2$. Row 3 minus $\tfrac12\times$ row 1: $(0, -\tfrac12, \tfrac52 \mid 2)$. Pivot 2 is $1$. Row 3 plus $\tfrac12\times$ row 2: $(0, 0, \tfrac72 \mid \tfrac72)$. Pivot 3 is $\tfrac72$. Back substitution: $z = 1$, $y = 3 - 2z = 1$, $2x = 4 - y - z = 2$, so $x = 1$. The solution is $(1, 1, 1)$, with pivots $2$, $1$, $\tfrac72$.

## 6. A singular system

- Row 2 minus $2\times$ row 1: $(0, 1, 2 \mid 1)$.
- Row 3 minus $3\times$ row 1: $(0, 1, 2 \mid b_3 - 12)$.
- Row 3 minus row 2: $(0, 0, 0 \mid b_3 - 13)$.

There is no pivot in column 3. The last equation reads $0 = b_3 - 13$. It holds only for $b_3 = 13$, and then there are infinitely many solutions. For any other $b_3$ it is a contradiction such as $0 = 1$, so there is no solution. For $b_3 = 13$, pick $z = 0$: then $y = 1$ and $x = 4 - 2y - z = 2$, so $(2, 1, 0)$ is a solution (check: $2 + 2 = 4$, $4 + 5 = 9$, $6 + 7 = 13$). Setting $z$ to a different number gives a different solution.

## 7. Elementary matrices

The steps were: row 2 minus $2\times$ row 1, row 3 minus $1\times$ row 1, and row 3 plus $2\times$ row 2. The matrices put $-\ell$ in row $i$, column $j$:

```math
E_{21} = \begin{bmatrix} 1 & 0 & 0 \\ -2 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix},\quad
E_{31} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ -1 & 0 & 1 \end{bmatrix},\quad
E_{32} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 2 & 1 \end{bmatrix}.
```

The product $E_{32}E_{31}E_{21}A$ should equal $U$, with the first step nearest $A$. The permutation matrix that swaps rows 1 and 2 is $P_{12}$ with rows $(0,1,0)$, $(1,0,0)$, $(0,0,1)$.

```python
import numpy as np

A = np.array([[1, 1, 1], [2, 3, 1], [1, -1, 2]])
b = np.array([2, 2, 5])
E21 = np.array([[1, 0, 0], [-2, 1, 0], [0, 0, 1]])
E31 = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 1]])
E32 = np.array([[1, 0, 0], [0, 1, 0], [0, 2, 1]])
print(E32 @ E31 @ E21 @ A)
print(E32 @ E31 @ E21 @ b)

A5 = np.array([[0, 1, 2], [2, 1, 1], [1, 0, 3]])
P12 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
print(P12 @ A5)
```

```text
[[ 1  1  1]
 [ 0  1 -1]
 [ 0  0 -1]]
[ 2 -2 -1]
[[2 1 1]
 [0 1 2]
 [1 0 3]]
```

The first output is the triangular system of problem 4, and the second is its right-hand side $\mathbf{c} = (2, -2, -1)$. The last output shows that $P_{12}A$ is $A$ with its first two rows exchanged, which is the first step of problem 5.

## 8. Fit a curve

Each point gives an equation: at $t = 1$, $a + b + c = 6$; at $t = 2$, $a + 2b + 4c = 11$; at $t = 3$, $a + 3b + 9c = 18$.

```math
\begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 4 \\ 1 & 3 & 9 \end{bmatrix}
\begin{bmatrix} a \\ b \\ c \end{bmatrix}
=
\begin{bmatrix} 6 \\ 11 \\ 18 \end{bmatrix}
```

Eliminate: row 2 minus row 1 gives $b + 3c = 5$, and row 3 minus row 1 gives $2b + 8c = 12$. Subtract $2\times$ the first of these from the second: $2c = 2$, so $c = 1$. Then $b = 5 - 3 = 2$ and $a = 6 - 2 - 1 = 3$. The curve is $y = 3 + 2t + t^2$.

```python
V = np.vander([1, 2, 3], increasing=True)
print(V)
print(np.linalg.solve(V, [6, 11, 18]))
```

```text
[[1 1 1]
 [1 2 4]
 [1 3 9]]
[3. 2. 1.]
```

## 9. Cost

The work grows like $n^3$. Doubling $n$ from 500 to 1000 multiplies it by $2^3 = 8$, so about $16$ seconds. Going to $n = 1500$ multiplies it by $3^3 = 27$, so about $54$ seconds. For $n = 1000$, the number of multiplications is about $n^3/3 = 10^9/3 \approx 3.3 \times 10^8$, roughly 333 million.

## 10. Write a solver

```python
def solve_gauss(A, b):
    """Solve A x = b by elimination with row exchanges, then back substitution."""
    U = np.array(A, dtype=float)
    c = np.array(b, dtype=float)
    n = len(c)
    for k in range(n):
        p = k + np.argmax(np.abs(U[k:, k]))          # row with the largest entry in column k
        if abs(U[p, k]) < 1e-12:
            raise ValueError(f"no pivot in column {k}: the matrix is singular")
        if p != k:
            U[[k, p]] = U[[p, k]]
            c[[k, p]] = c[[p, k]]
        for i in range(k + 1, n):
            multiplier = U[i, k] / U[k, k]
            U[i, k:] -= multiplier * U[k, k:]
            c[i] -= multiplier * c[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


print(solve_gauss(A, b))                                     # problem 4
print(solve_gauss(A5, [3, 4, 4]))                            # problem 5

rng = np.random.default_rng(0)
A6 = rng.normal(size=(6, 6))
b6 = rng.normal(size=6)
print(np.max(np.abs(solve_gauss(A6, b6) - np.linalg.solve(A6, b6))) < 1e-10)

A_sing = np.array([[1, 2, 1], [2, 5, 4], [3, 7, 5]])
try:
    solve_gauss(A_sing, [4, 9, 13])
except ValueError as error:
    print("ValueError:", error)
```

```text
[ 2. -1.  1.]
[1. 1. 1.]
True
ValueError: no pivot in column 2: the matrix is singular
```

The solver agrees with NumPy on the two hand problems and on a random system. On the singular matrix it raises an error even though $b_3 = 13$ gives a consistent system. That is reasonable: for this matrix there are infinitely many solutions, so there is no single answer to return. A function that returns one vector has to refuse, and a deeper treatment (finding all the solutions) is the topic of [vector spaces and subspaces](../../vector-spaces-and-subspaces/README.md).
