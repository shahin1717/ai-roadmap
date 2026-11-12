# Exercises: Projections and Least Squares

Ten problems, easy to harder. Do the first nine on paper, then check yourself in the [notebook](../notebook.ipynb) or with NumPy. The last one is a coding task. Worked answers are in [solutions.md](solutions.md). Give each problem a real attempt before you open them.

Back to the [topic](../README.md).

## Projection onto a line

**1. Project.** Let $\mathbf{b} = (1, 2, 3)$. (a) Project $\mathbf{b}$ onto the line through $\mathbf{a} = (1, 1, 1)$ and find the error. (b) Project $\mathbf{b}$ onto the line through $\mathbf{a} = (2, -1, 2)$. In both cases check that the error is perpendicular to $\mathbf{a}$.

**2. A projection matrix for a line.** Write the projection matrix $P$ onto the line through $\mathbf{a} = (1, 2)$. Check that $P^2 = P$ and $P^\top = P$, find the trace of $P$, and compute $P\mathbf{b}$ for $\mathbf{b} = (4, 2)$.

## Projection onto a subspace

**3. Normal equations.** Let the columns of $A$ be $(1, 0, 1)$ and $(0, 1, 1)$, and let $\mathbf{b} = (1, 2, 4)$. (a) Write $A^\top A$ and $A^\top\mathbf{b}$, and solve the normal equations for $\hat{\mathbf{x}}$. (b) Find $\mathbf{p}$ and $\mathbf{e}$, and check that $A^\top\mathbf{e} = \mathbf{0}$. (c) Is $\mathbf{b}$ in the column space? How can you tell from $\mathbf{e}$?

**4. The projection matrix.** For the matrix $A$ of problem 3, compute $P = A(A^\top A)^{-1}A^\top$. Check that $P\mathbf{b} = \mathbf{p}$, that $P^2 = P$, and that $\operatorname{trace}(P) = 2$. Then compute $I - P$: what is its trace, and what subspace does it project onto?

**5. Why $I - P$ is a projection too.** Show algebraically that $(I - P)^2 = I - P$ and $(I - P)^\top = I - P$ for any projection matrix $P$. Then show that $P(I - P) = 0$ and explain in words what that says about projecting the error again.

## Least squares

**6. A line fit by hand.** Fit $y = c + dt$ to the points $(1, 2)$, $(2, 3)$, $(3, 5)$, $(4, 4)$. Set up the design matrix, solve the normal equations, and find the fitted values and residuals. Check that the residuals sum to zero and are perpendicular to the inputs $t$, and compute the sum of squared residuals.

**7. Overfitting.** For the same four points, a cubic $y = a + bt + ct^2 + dt^3$ passes through all of them. Find its coefficients (use a computer for the solve), and compare the cubic's prediction at $t = 5$ with the line's prediction. Which would you trust more, and why?

**8. Dependent columns.** The matrix $X$ with rows $(1,2,3)$, $(2,0,2)$, $(0,1,1)$, $(3,1,4)$ has a third column equal to the sum of the first two. For $\mathbf{y} = (6, 4, 2, 8)$, show that $(2, 2, 0)$ solves $X\mathbf{w} = \mathbf{y}$ exactly, and then find **all** solutions. Among them, find the one of smallest length (the minimum-norm solution), and say which NumPy function returns it.

## Projecting out a direction, and code

**9. Remove a direction.** Take the unit vector $\mathbf{d} = (1, 1)/\sqrt2$. (a) Remove the component along $\mathbf{d}$ from $\mathbf{x} = (3, 4)$, and check that the result is perpendicular to $\mathbf{d}$. (b) Write the matrix $I - \mathbf{d}\mathbf{d}^\top$ that does this to every point.

**10. Write and test.** Write `least_squares(A, b)` that solves a least-squares problem with QR (`np.linalg.qr`, then back substitution on $R\mathbf{x} = Q^\top\mathbf{b}$). (a) Test it against `np.linalg.lstsq` on a random $50 \times 4$ matrix with noisy targets (use a seeded generator), and check that the residual is perpendicular to every column. (b) Fit polynomials of degrees 1 through 8 to 15 noisy samples of a sine wave, and print the training error and the error on 200 fresh points for each degree. Which degree has the best test error?
