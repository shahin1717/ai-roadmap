# Exercises: Linear Transformations

Ten problems, easy to harder. Do the first nine on paper, then check yourself in the [notebook](../notebook.ipynb) or with NumPy. The last one is a coding task. Worked answers are in [solutions.md](solutions.md). Give each problem a real attempt before you open them.

Back to the [topic](../README.md).

## Is it linear, and what follows from it

**1. Linear or not?** Decide which of these maps are linear, and justify each answer with the two rules or with a single failing example. (a) $T(x, y) = (x - y,\ 3x)$. (b) $T(x, y) = (x + 1,\ y)$. (c) $T(x, y) = (xy,\ x)$. (d) $T(x, y, z) = (z,\ x + y + z)$, which goes from $\mathbb{R}^3$ to $\mathbb{R}^2$. For each linear one, write its matrix.

**2. Use linearity.** A linear map $T$ on the plane has $T(1, 1) = (3, 1)$ and $T(1, -1) = (1, 5)$. Find $T(1, 0)$ and $T(0, 1)$ by writing $\mathbf{e}_1$ and $\mathbf{e}_2$ as combinations of $(1, 1)$ and $(1, -1)$. Then write the matrix of $T$ and compute $T(5, -2)$.

## The matrix of a transformation

**3. Matrices from pictures.** Write the matrix of each linear map by asking where $\mathbf{e}_1$ and $\mathbf{e}_2$ go. (a) Reflection in the line $y = -x$. (b) Rotation by $45^\circ$. (c) The shear that slides every point up by half of its $x$-coordinate, $T(x, y) = (x,\ y + x/2)$.

**4. Order matters.** Let $R$ be the quarter turn (counterclockwise) and $P$ the projection onto the $x$-axis. Compute $PR$ and $RP$. Apply each to $(2, 3)$ and describe in words what each product does. Find the kernel and range of each.

## Kernel, range, and other vector spaces

**5. Kernel and range.** Let $A$ have rows $(1, 2, 3)$, $(0, 1, 1)$, $(1, 3, 4)$ and $T(\mathbf{v}) = A\mathbf{v}$. Find a basis for the kernel and one for the range, and check that their dimensions add up to 3. Is $T$ invertible? Is $(1, 1, 2)$ in the range? Is $(1, 1, 3)$?

**6. Differentiate and integrate.** Use the bases $1, x, x^2$ for $\mathcal{P}_2$ and $1, x, x^2, x^3$ for $\mathcal{P}_3$. Let $J: \mathcal{P}_2 \to \mathcal{P}_3$ send $p$ to $\int_0^x p(t)\,dt$, and let $D: \mathcal{P}_3 \to \mathcal{P}_2$ be the derivative (a cubic's derivative has degree at most 2). (a) Write the matrices of $J$ (size $4 \times 3$) and $D$ (size $3 \times 4$). (b) Compute $DJ$ and $JD$, say what size each has, and explain in words why one is the identity and the other is not.

## Changing the basis

**7. A diagonalizing basis.** Let $A$ have rows $(4, 1)$ and $(2, 3)$, and let the new basis be $\mathbf{m}_1 = (1, 1)$, $\mathbf{m}_2 = (1, -2)$. (a) Compute $A\mathbf{m}_1$ and $A\mathbf{m}_2$ and write each as a multiple of $\mathbf{m}_1$ or $\mathbf{m}_2$. What is $B = M^{-1}AM$ without any multiplication? (b) Use $A = MBM^{-1}$ to compute $A^3$, and check it by multiplying $A$ out.

**8. Similar matrices.** (a) For $A$ with rows $(1, 2)$ and $(3, 4)$ and $M$ with rows $(1, 1)$ and $(0, 1)$, compute $B = M^{-1}AM$ and check that $A$ and $B$ have the same trace and determinant. (b) The identity $I$ and the shear $S$ with rows $(1, 1)$ and $(0, 1)$ have the same trace and the same determinant. Are they similar? Compute $M^{-1}IM$ for any invertible $M$ and decide.

## Homogeneous coordinates and code

**9. Scale about a point.** Write the $3 \times 3$ matrix that scales the plane by a factor 2 about the point $(1, 1)$ (so $(1, 1)$ stays where it is). Build it as three steps: shift $(1, 1)$ to the origin, scale, shift back. Apply it to $(3, 1)$ and check that $(1, 1)$ is fixed.

**10. Write and test.** Write `matrix_of(T, n)` that builds the matrix of a linear map $T:\mathbb{R}^n \to \mathbb{R}^m$ by applying `T` to each standard basis vector. (a) Test it on a rotation by $25^\circ$ written as a plain function, and on a map defined by a hidden random matrix (check that you recover it with `np.allclose`). (b) Apply it to the function $\mathbf{x} \mapsto$ `np.convolve(x, [1, 2, 3], mode="same")` with $n = 6$, and print the matrix. What pattern does it have? (c) Show that the same function applied to a nonlinear map such as $\mathbf{x} \mapsto \mathbf{x}^2$ produces a matrix that does **not** reproduce the map.
