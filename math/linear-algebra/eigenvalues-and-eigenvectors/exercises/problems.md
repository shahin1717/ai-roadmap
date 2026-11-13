# Exercises: Eigenvalues and Eigenvectors

Ten problems, easy to harder. Do the first nine on paper, then check yourself in the [notebook](../notebook.ipynb) or with NumPy. The last one is a coding task. Worked answers are in [solutions.md](solutions.md). Give each problem a real attempt before you open them.

Back to the [topic](../README.md).

## Eigenvalues and eigenvectors

**1. Is it an eigenvector?** Let $`A = \begin{bmatrix} 4 & -1 \\ 2 & 1 \end{bmatrix}`$. For each of $(1,2)$, $(1,1)$, and $(1,0)$, decide whether it is an eigenvector of $A$, and if so give the eigenvalue.

**2. The characteristic equation.** For $`A = \begin{bmatrix} 5 & 2 \\ 2 & 2 \end{bmatrix}`$, write $\det(A - \lambda I)$ as a polynomial, find the eigenvalues, and then find an eigenvector for each as the nullspace of $A - \lambda I$. Check that the two eigenvectors are orthogonal and explain why that was to be expected.

**3. A triangular matrix.** The matrix $A$ has rows $(3,1,2)$, $(0,5,4)$, $(0,0,1)$. Read off its eigenvalues, and check them against the trace and the determinant. Is $A$ invertible? What are the eigenvalues of $A^2$, of $A^{-1}$, and of $A + 2I$?

**4. Using the properties.** A $3 \times 3$ matrix $A$ has eigenvalues $1$, $2$, and $4$. Without finding $A$, give $\det A$, $\operatorname{tr} A$, the eigenvalues of $A^2 + I$, of $A^{-1}$, and of $A + 3I$. Is $A$ singular?

**5. Projections, reflections, rotations.** (a) The projection matrix onto the line through $(3,4)$ is $`P = \begin{bmatrix} 0.36 & 0.48 \\ 0.48 & 0.64 \end{bmatrix}`$. Find its eigenvalues and eigenvectors from the geometry alone (which vectors are unchanged, and which are sent to zero?), and check one with $P\mathbf{x}$. (b) What are the eigenvalues and eigenvectors of the reflection with rows $(1,0)$ and $(0,-1)$? (c) Why does a rotation by $90^\circ$ have no real eigenvector, and what are its eigenvalues?

## Diagonalization and symmetric matrices

**6. Diagonalize.** Let $`A = \begin{bmatrix} 5 & 4 \\ 1 & 2 \end{bmatrix}`$. Find its eigenvalues and eigenvectors, write $S$ and $\Lambda$, and use $A = S\Lambda S^{-1}$ to compute $A^3$. Check against direct multiplication.

**7. A long-run state.** A population moves between two regions each year: $`M = \begin{bmatrix} 0.9 & 0.5 \\ 0.1 & 0.5 \end{bmatrix}`$ takes this year's shares $(x, y)$ to next year's. Find the eigenvalues of $M$, the eigenvector for the eigenvalue $1$ scaled so its entries add to $1$, and say where the shares end up after many years, starting from $(0, 1)$. Why does the other eigenvalue not matter in the long run?

**8. A defective matrix.** Find the eigenvalues and eigenvectors of $`\begin{bmatrix} 3 & 1 \\ 0 & 3 \end{bmatrix}`$. Is it diagonalizable? Compare with $`\begin{bmatrix} 3 & 0 \\ 0 & 3 \end{bmatrix}`$, which has the same eigenvalues.

**9. The spectral theorem.** For $`C = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}`$, find $Q$ and $\Lambda$ with $C = Q\Lambda Q^\top$, and write $C$ as $\lambda_1\mathbf{q}_1\mathbf{q}_1^\top + \lambda_2\mathbf{q}_2\mathbf{q}_2^\top$. Is $C$ positive definite? Evaluate $\mathbf{x}^\top C\mathbf{x}$ for $\mathbf{x} = (1,-1)$. Then decide whether $`\begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}`$ is positive definite.

## Code

**10. Power method and step size.** (a) Write `power_method(A, v0, steps)` that multiplies by $A$ and normalizes, and returns the final vector and the Rayleigh quotient. Run it on $`\begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}`$ and show that the error shrinks by about $|\lambda_2/\lambda_1| = 0.4$ per step. (b) For the quadratic $f(\mathbf{x}) = \tfrac12\mathbf{x}^\top H\mathbf{x}$ with $H = \operatorname{diag}(1, 10)$, run gradient descent $\mathbf{x} \leftarrow \mathbf{x} - \alpha H\mathbf{x}$ for 200 steps from $(1,1)$ with $\alpha = 0.19$ and $\alpha = 0.21$, and relate the outcomes to $2/\lambda_{\max}$.
