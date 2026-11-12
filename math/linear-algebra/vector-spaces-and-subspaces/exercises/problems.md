# Exercises: Vector Spaces and Subspaces

Ten problems, easy to harder. Do the first nine on paper, then check yourself in the [notebook](../notebook.ipynb) or with NumPy. The last one is a coding task. Worked answers are in [solutions.md](solutions.md). Give each problem a real attempt before you open them.

Back to the [topic](../README.md).

## Subspaces and the column space

**1. Subspace or not?** For each set, decide whether it is a subspace, and give the reason (a rule that fails, or why both hold). (a) $\{(x, y) : y = 3x\}$ in $\mathbb{R}^2$. (b) $\{(x, y) : x + y = 1\}$ in $\mathbb{R}^2$. (c) $\{(x, y) : x \ge 0 \text{ and } y \ge 0\}$ in $\mathbb{R}^2$. (d) $\{(x, y, z) : x = y\}$ in $\mathbb{R}^3$. (e) $\{(x, y) : xy = 0\}$ in $\mathbb{R}^2$.

**2. A column space.** Let $`A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}`$. (a) Describe $C(A)$ as the set of vectors $(b_1, b_2, b_3)$ that satisfy one equation. (b) Decide whether $A\mathbf{x} = (2, 3, 5)$ and $A\mathbf{x} = (1, 1, 3)$ are solvable. (c) What does elimination on $[A \mid \mathbf{b}]$ look like in the unsolvable case?

## The nullspace

**3. By inspection.** Find the nullspace of $`\begin{bmatrix} 1 & 2 \\ 3 & 6 \end{bmatrix}`$ and describe it geometrically. Then explain why the nullspace of $`\begin{bmatrix} 1 & 2 \\ 3 & 7 \end{bmatrix}`$ is only $\{\mathbf{0}\}$.

**4. Row reduce.** Reduce
```math
A = \begin{bmatrix} 1 & 2 & 1 & 3 \\ 2 & 4 & 3 & 8 \\ 3 & 6 & 4 & 11 \end{bmatrix}
```
to its reduced row echelon form $R$. Name the pivot columns and the free columns, and give the rank.

**5. Special solutions.** For the matrix of problem 4, find the special solutions, check that each satisfies $A\mathbf{s} = \mathbf{0}$, and describe $N(A)$.

**6. The complete solution.** For the same $A$, find all solutions of $A\mathbf{x} = (1, 3, 4)$ by row reducing $[A \mid \mathbf{b}]$ and writing $\mathbf{x}_p + \mathbf{x}_n$. Then show that $A\mathbf{x} = (1, 3, 5)$ has no solution.

## Cases, geometry, and code

**7. Four cases.** For each matrix, state $m$, $n$, the rank $r$, and how many solutions $A\mathbf{x} = \mathbf{b}$ can have (depending on $\mathbf{b}$). (a) The matrix with rows $(1,2,3)$ and $(0,1,4)$. (b) The matrix of problem 2. (c) The matrix with rows $(1,2)$ and $(2,4)$. (d) The matrix with rows $(2,1)$ and $(1,1)$.

**8. A line in space.** Find the nullspace of the $2 \times 3$ matrix with rows $(1,1,1)$ and $(1,2,3)$. Describe it as a line in $\mathbb{R}^3$ and say what it means geometrically for the two equations $x + y + z = 0$ and $x + 2y + 3z = 0$.

**9. Shifted solution sets.** For the matrix with rows $(1,2)$ and $(2,4)$ and $\mathbf{b} = (5, 10)$, check that $(1, 2)$ and $(3, 1)$ both solve $A\mathbf{x} = \mathbf{b}$. Show that their difference lies in the nullspace, and write the complete solution twice: once using $\mathbf{x}_p = (5, 0)$ and once using $\mathbf{x}_p = (1, 2)$. Why do both describe the same set?

**10. Code it.** Write a function `rref(A)` that returns the reduced row echelon form and the list of pivot columns, and a function `special_solutions(R, pivots)` that returns the special solutions as columns of a matrix. Test them on (a) the matrix of problem 4, comparing with `sympy.Matrix(A).rref()`, and (b) a random $5 \times 6$ matrix of rank 2 built as a product of a $5 \times 2$ and a $2 \times 6$ matrix with a seeded generator. For (b), check that the rank is 2, that there are $6 - 2 = 4$ special solutions, and that $A$ times each one is zero.
