# Exercises: Linear Equations and Gaussian Elimination

Ten problems, easy to harder. Do the first nine on paper, then check yourself in the [notebook](../notebook.ipynb) or with NumPy. The last one is a coding task. Worked answers are in [solutions.md](solutions.md). Give each problem a real attempt before you open them.

Back to the [topic](../README.md).

## Warm-up

**1. Write it as $A\mathbf{x} = \mathbf{b}$.** Write the system $2x + y = 7$, $x - 3y = -7$ in the form $A\mathbf{x} = \mathbf{b}$. Check that $(x, y) = (2, 3)$ is a solution, and say which combination of the columns of $A$ it describes.

**2. How many solutions?** Decide whether each system has one solution, none, or infinitely many. (a) $x + y = 3$, $2x + 2y = 6$. (b) $x + y = 3$, $2x + 2y = 7$. (c) $x - y = 1$, $x + y = 5$. For the systems that have solutions, give one.

## Elimination by hand

**3. A 2 by 2 system.** Solve $2x + 3y = 8$, $4x - y = 2$ by elimination. State the multiplier and the two pivots, then use back substitution.

**4. A 3 by 3 system.** Solve
$$
x + y + z = 2, \qquad 2x + 3y + z = 2, \qquad x - y + 2z = 5
$$
by elimination. Write down each multiplier, the triangular system you reach, and the three pivots.

**5. A zero pivot.** Solve
$$
y + 2z = 3, \qquad 2x + y + z = 4, \qquad x + 3z = 4.
$$
Which step is needed first, and why? What are the pivots after it?

**6. A singular system.** Consider
$$
x + 2y + z = 4, \qquad 2x + 5y + 4z = 9, \qquad 3x + 7y + 5z = b_3.
$$
Eliminate, and find the one value of $b_3$ for which the system has solutions. For that value, give one solution. What happens for any other $b_3$?

## Elimination with matrices, and applications

**7. Elementary matrices.** For the system in problem 4, write the elementary matrices $E_{21}$, $E_{31}$, $E_{32}$ that carry out the three steps. Multiply them with $A$ in the correct order and check that you get the triangular $U$ from problem 4. Then write the permutation matrix $P_{12}$ for a 3 by 3 system and compute $P_{12}A$ for the matrix $A$ of problem 5.

**8. Fit a curve.** A quadratic $y = a + bt + ct^2$ passes through the points $(1, 6)$, $(2, 11)$, and $(3, 18)$. Write the three equations for $a$, $b$, $c$, set them up as $A\mathbf{x} = \mathbf{b}$, and solve.

**9. Cost.** A solver takes about 2 seconds for a system with $n = 500$ equations. Assuming the time is proportional to $n^3$, estimate the time for $n = 1000$ and for $n = 1500$. Roughly how many multiplications does elimination use for $n = 1000$?

## Code

**10. Write a solver.** Write a function `solve_gauss(A, b)` that solves $A\mathbf{x} = \mathbf{b}$ using forward elimination with row exchanges (swap in the row with the largest entry in the column), followed by back substitution. It should raise an error if it finds no pivot. Test it on problems 4 and 5, and on a random 6 by 6 system made with a seeded generator, by comparing with `numpy.linalg.solve`. Finally, run it on the matrix of problem 6 with $b_3 = 13$. That system has solutions, yet your function raises an error. Why is that reasonable for a solver that is meant to return a single answer?
