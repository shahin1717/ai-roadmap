# Exercises: Orthogonality and Gram-Schmidt

Ten problems, easy to harder. Do the first nine on paper, then check yourself in the [notebook](../notebook.ipynb) or with NumPy. The last one is a coding task. Worked answers are in [solutions.md](solutions.md). Give each problem a real attempt before you open them.

Back to the [topic](../README.md).

## Orthogonal vectors and subspaces

**1. Orthogonal or not?** (a) Are $(1,2,2)$ and $(2,1,-2)$ orthogonal? (b) Are $(1,1)$ and $(1,-2)$ orthogonal? (c) For the pair that is orthogonal, check Pythagoras: compare $\|\mathbf{v} + \mathbf{w}\|^2$ with $\|\mathbf{v}\|^2 + \|\mathbf{w}\|^2$.

**2. A complement.** Let $V$ be the line through $(1,2,2)$ in $\mathbb{R}^3$. Describe $V^\perp$ as the solutions of one equation, find a basis for it, and state its dimension. Check that each basis vector is orthogonal to $(1,2,2)$.

**3. Four subspaces, two orthogonal pairs.** Let
```math
A = \begin{bmatrix} 1 & 2 & 0 \\ 0 & 0 & 1 \\ 1 & 2 & 1 \end{bmatrix}.
```
Find a basis for each of the four fundamental subspaces. Then verify that the row space is orthogonal to the nullspace and the column space to the left nullspace, and that the dimensions of each pair add up to the dimension of the space they live in.

**4. Splitting a vector.** The $2 \times 3$ matrix $B$ with rows $(1,1,1)$ and $(1,2,3)$ has nullspace spanned by $(1,-2,1)$. Split $\mathbf{x} = (1,-1,2)$ as $\mathbf{x}_r + \mathbf{x}_n$ with $\mathbf{x}_n$ in the nullspace and $\mathbf{x}_r$ in the row space. (Hint: $\mathbf{x}_n$ is a multiple of $(1,-2,1)$ chosen so that what remains is orthogonal to it.) Check that $\mathbf{x}_r$ really is in the row space and that the two parts are orthogonal.

## Orthonormal bases and orthogonal matrices

**5. Orthonormal basis.** (a) Show that $\mathbf{q}_1 = \tfrac13(1,2,2)$ and $\mathbf{q}_2 = \tfrac13(2,1,-2)$ are orthonormal. (b) Check that $\mathbf{q}_3 = \tfrac13(2,-2,1)$ completes them to an orthonormal basis of $\mathbb{R}^3$. (c) Find the coordinates of $\mathbf{v} = (3,3,0)$ in this basis using dot products only.

**6. Orthogonal matrices.** (a) Show that $`Q = \begin{bmatrix} 0.6 & 0.8 \\ -0.8 & 0.6 \end{bmatrix}`$ is orthogonal, and write down $Q^{-1}$ without doing any elimination. (b) Rotate $(3,4)$ by $90^\circ$ counterclockwise, using the matrix with rows $(0,-1)$ and $(1,0)$, and compare the lengths. (c) Check that $`\begin{bmatrix} 0.6 & 0.8 \\ 0.8 & -0.6 \end{bmatrix}`$ is orthogonal too, and find its determinant. What kind of map is it?

## Gram-Schmidt and QR

**7. A friendly case.** Apply Gram-Schmidt to $\mathbf{a}_1 = (2,0,0)$, $\mathbf{a}_2 = (1,3,0)$, $\mathbf{a}_3 = (1,2,4)$, and write $A = QR$. What do you notice about $Q$ and $R$?

**8. Three vectors.** Apply Gram-Schmidt to $\mathbf{a}_1 = (1,1,1)$, $\mathbf{a}_2 = (1,2,3)$, $\mathbf{a}_3 = (1,0,0)$. Give $\mathbf{q}_1$, $\mathbf{q}_2$, $\mathbf{q}_3$ and the matrix $R$, and check that $\mathbf{q}_3$ is perpendicular to the other two.

**9. Solving with QR.** For $\mathbf{a}_1 = (3,4)$ and $\mathbf{a}_2 = (1,7)$, the factorization $A = QR$ has $Q$ with columns $(0.6, 0.8)$ and $(-0.8, 0.6)$ and $R$ with rows $(5, 6.2)$ and $(0, 3.4)$ (you can verify this). Use it to solve $A\mathbf{x} = (4, 11)$ in two steps: compute $Q^\top\mathbf{b}$, then solve $R\mathbf{x} = Q^\top\mathbf{b}$ by back substitution.

## Code

**10. Write and test.** Write `gram_schmidt(A)` (classical) and `modified_gram_schmidt(A)`. (a) Check both on the matrix of problem 8: $A = QR$ and $Q^\top Q = I$, using `np.allclose`. (b) Compare their orthogonality error $\|Q^\top Q - I\|$ on the $6 \times 6$ and $8 \times 8$ Hilbert matrices (`scipy.linalg.hilbert`), together with `np.linalg.qr`. (c) For a seeded random pair of vectors in 50 dimensions, repeated 5000 times, measure the standard deviation of the cosine of the angle and compare it with $1/\sqrt{50}$.
