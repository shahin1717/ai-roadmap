# Exercises: Matrix Operations and LU Decomposition

Ten problems, easy to harder. Do the first nine on paper, then check yourself in the [notebook](../notebook.ipynb) or with NumPy. The last one is a coding task. Worked answers are in [solutions.md](solutions.md). Give each problem a real attempt before you open them.

Back to the [topic](../README.md).

## Rules and products

**1. Two ways to multiply.** Let
```math
A = \begin{bmatrix} 2 & 0 \\ 1 & 3 \end{bmatrix}, \qquad B = \begin{bmatrix} 1 & 4 \\ 2 & 0 \end{bmatrix}.
```
(a) Compute $AB$ entry by entry, using dot products. (b) Check the first column of $AB$ as $A$ times the first column of $B$. (c) Compute $BA$. Are $AB$ and $BA$ equal?

**2. True or false?** For square matrices $A$, $B$, $C$ of the same size, decide whether each statement is always true. For each false one, give a small counterexample. (a) $(AB)C = A(BC)$. (b) $AB = BA$. (c) $A(B + C) = AB + AC$. (d) $(A + B)^2 = A^2 + 2AB + B^2$. (e) If $AB = 0$ then $A = 0$ or $B = 0$.

**3. Powers, and two layers in one.** (a) For $`A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}`$ compute $A^2$ and $A^3$, and guess a formula for $A^n$. (b) Let $W_1$ have rows $(1, 0, 2)$ and $(0, 1, 1)$, and let $W_2$ have rows $(1, 1)$, $(2, 0)$, and $(0, 3)$. For $\mathbf{x} = (1, 2, 3)$, compute $W_2(W_1\mathbf{x})$, then compute $W_2W_1$ and apply it to $\mathbf{x}$. What do you notice about the shapes and the answers?

## Inverses and determinants

**4. A 2 by 2 inverse.** (a) Find $\det A$ for $`A = \begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix}`$. (b) Use the formula that swaps the diagonal entries, negates the off-diagonal ones, and divides by $\det A$ to get $A^{-1}$, and check that $AA^{-1} = I$. (c) Use $A^{-1}$ to solve $A\mathbf{x} = (1, 0)$. (d) Why does $`\begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}`$ have no inverse?

**5. Gauss-Jordan.** Find the inverse of
```math
A = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 1 & 4 \\ 0 & 0 & 1 \end{bmatrix}
```
by row operations on $[A \mid I]$, and verify it by multiplying.

**6. Determinants.** Find (a) $`\det \begin{bmatrix} 3 & 1 \\ 2 & 4 \end{bmatrix}`$. (b) The determinant of the $3 \times 3$ matrix with rows $(2,1,-1)$, $(4,3,1)$, $(-2,2,5)$, from its pivots. (c) The determinant of the matrix with rows $(0,2,1)$, $(1,1,1)$, $(2,1,3)$, which needs a row exchange. (d) For $`A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}`$ and $`B = \begin{bmatrix} 0 & 1 \\ 1 & 2 \end{bmatrix}`$, check that $\det(AB) = \det A\,\det B$.

## LU, transposes, and code

**7. LU by hand.** Factor the matrix with rows $(1,2,1)$, $(3,8,1)$, $(0,4,1)$ as $A = LU$ without row exchanges. List the multipliers, write $L$ and $U$, and check that $LU = A$.

**8. Many right-hand sides.** Using the $L$ and $U$ from problem 7, solve $A\mathbf{x} = \mathbf{b}$ for $\mathbf{b} = (2, 12, 2)$ and for $\mathbf{b} = (1, 0, 0)$, each time by forward and back substitution. For a $1000 \times 1000$ system, roughly how many multiplications does the factorization take, and how many does each extra solve take?

**9. Transposes and permutations.** (a) For $`A = \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix}`$ and $`B = \begin{bmatrix} 3 & 0 \\ 1 & 2 \end{bmatrix}`$, check that $(AB)^\top = B^\top A^\top$. (b) Compute $A^\top A$ for the matrix with rows $(1,2)$, $(3,4)$, $(5,6)$ and explain why the result is symmetric. (c) Let $P$ have rows $(0,1,0)$, $(0,0,1)$, $(1,0,0)$. Compute $P\mathbf{v}$ for $\mathbf{v} = (7, 8, 9)$, and show that $P^\top P = I$.

**10. Write an LU routine.** Write a function `lu_decompose(A)` that returns $P$, $L$, $U$ with $A = PLU$, using partial pivoting (swap in the largest entry of the column). Check it against `scipy.linalg.lu` on a random 6 by 6 matrix made with a seeded generator. Then (a) compute $\det A$ from $U$ and the sign of $P$, and compare with `numpy.linalg.det`; (b) solve for 20 different right-hand sides with one call to `scipy.linalg.lu_factor` and 20 calls to `lu_solve`, and compare with `numpy.linalg.solve`.
