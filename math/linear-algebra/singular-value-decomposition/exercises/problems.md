# Exercises: Singular Value Decomposition

Ten problems, easy to harder. Do the first nine on paper, then check yourself in the [notebook](../notebook.ipynb) or with NumPy. The last one is a coding task. Worked answers are in [solutions.md](solutions.md). Give each problem a real attempt before you open them.

Back to the [topic](../README.md).

## The geometry and the factorization

**1. Reading the ellipse.** Let $`A = \begin{bmatrix} 3 & 0 \\ 0 & -2 \end{bmatrix}`$ and $`B = \begin{bmatrix} 0 & -2 \\ 1 & 0 \end{bmatrix}`$. For each matrix, find $A^\top A$ (or $B^\top B$), its eigenvalues, and the singular values. Then give the half-axes of the ellipse that the matrix makes from the unit circle, and compare the product of the singular values with $|\det|$.

**2. A full SVD by hand.** Let $`A = \begin{bmatrix} 2 & 2 \\ -1 & 1 \end{bmatrix}`$. Find $A^\top A$, its eigenvalues and unit eigenvectors, then $\sigma_1, \sigma_2$, $V$, and $U$ (using $\mathbf{u}_i = A\mathbf{v}_i/\sigma_i$). Write out $A = U\Sigma V^\top$ and check it. What is $|\det A|$?

**3. A rank-one matrix.** Let $`C = \begin{bmatrix} 1 & 1 & 1 \\ 2 & 2 & 2 \end{bmatrix}`$. (a) What is its rank, and what is its only nonzero singular value? (b) Give $\mathbf{u}_1$ and $\mathbf{v}_1$. (c) Describe all four fundamental subspaces, and say which of them $\mathbf{u}_2$ and $\mathbf{v}_2, \mathbf{v}_3$ belong to.

## Subspaces, rank, and conditioning

**4. Reading the singular values.** A $5 \times 4$ matrix has singular values $7,\ 3,\ 0.5,\ 0$. (a) What is its rank? (b) What are the dimensions of the column space, row space, nullspace, and left nullspace, and which columns of $U$ and $V$ give bases of them? (c) What is the condition number on the part where the matrix is invertible, $\sigma_1/\sigma_r$? (d) What fraction of the energy does the best rank-$2$ approximation keep, and what is its 2-norm error?

**5. Both Gram matrices.** Let $`A = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 1 \end{bmatrix}`$. (a) Find the eigenvalues of $A^\top A$ and of $AA^\top$. What do they have in common? (b) Give the singular values. (c) Which vector spans the left nullspace? Check it.

**6. The pseudoinverse.** Let $`B = \begin{bmatrix} 1 & 2 \\ 2 & 4 \\ 3 & 6 \end{bmatrix}`$ (rank $1$, with $\sigma_1 = \sqrt{70}$, $\mathbf{u}_1 = (1,2,3)/\sqrt{14}$, $\mathbf{v}_1 = (1,2)/\sqrt5$). (a) Write $B^+$. (b) Find the minimum-norm least-squares solution of $B\mathbf{x} \approx \mathbf{b}$ for $\mathbf{b} = (1, 2, 4)$. (c) Describe all least-squares solutions, and show that $\mathbf{x}^+$ is the shortest.

**7. An unstable system.** Let $`A = \begin{bmatrix} 1 & 1 \\ 1 & 1.01 \end{bmatrix}`$. (a) Solve $A\mathbf{x} = (2, 2.01)$ and $A\mathbf{x} = (2, 2.02)$ by hand. (b) Estimate the singular values using $\sigma_1\sigma_2 = |\det A|$ and $\sigma_1^2 + \sigma_2^2 = $ the sum of the squared entries, then the condition number. (c) Compare the relative change in $\mathbf{b}$ with the relative change in $\mathbf{x}$, and check it is below $\kappa$.

## Approximation and PCA

**8. The best rank-one approximation.** Use the matrix $A$ and the SVD from problem 2. Write $A_1 = \sigma_1\mathbf{u}_1\mathbf{v}_1^\top$ as a matrix. Compute $A - A_1$, and give its 2-norm and its Frobenius norm. What fraction of the energy does $A_1$ keep?

**9. Storage and choosing $k$.** A grayscale image is a $1000 \times 600$ matrix. (a) How many numbers does a rank-$k$ approximation store, and for which $k$ does it stop saving space? (b) Suppose the singular values are $\sigma_i = 50\cdot 0.8^{\,i-1}$ for $i = 1, \dots, 600$. What is the smallest $k$ that keeps at least $99\%$ of the energy, and what fraction of the original storage does that rank use?

**10. Code: PCA through the SVD.** Generate the data below, center it, and take the thin SVD. (a) Check that $\sigma_i^2/(n-1)$ equals the eigenvalues of `np.cov(X.T)` (sorted from largest) and that the rows of `Vt` match the eigenvectors up to sign. (b) Print the fraction of variance explained by each axis. (c) Keep only the first axis: build the rank-one reconstruction of the data (add the mean back) and check that the total squared error equals the sum of the dropped $\sigma_i^2$. (d) Run the same steps without centering, and describe how the first axis changes.

```python
import numpy as np
rng = np.random.default_rng(10)
X = rng.normal(size=(150, 3)) @ np.array([[2.0, 0.5, 0.0], [0.0, 1.0, 0.3], [0.0, 0.0, 0.2]]) + np.array([8.0, -5.0, 3.0])
```
