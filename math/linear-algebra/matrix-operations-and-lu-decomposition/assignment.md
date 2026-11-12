# Assignment: Heat in a Rod, Factored Once

Use an LU factorization to solve many versions of one physical problem with a single elimination. It takes about 45-60 minutes and needs only NumPy and Matplotlib. Back to the [topic](README.md).

## The setup

A thin rod has both ends held at $0$ degrees. We look at 8 evenly spaced points along it. If heat is added at the points (a vector $\mathbf{f}$ of 8 numbers), the steady temperatures $\mathbf{u}$ at those points satisfy a linear system $T\mathbf{u} = \mathbf{f}$, where $T$ has $2$ on the diagonal and $-1$ just above and just below it. The numbers are simplified for this exercise.

We want the temperature profile for five different heating scenarios. They all share the same matrix $T$, which is exactly the situation where you factor once and solve many times.

```python
import numpy as np

n = 8
T = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)

F = np.zeros((n, 5))      # one column per scenario
F[1, 0] = 1               # scenario 1: one unit of heat at point 2
F[4, 1] = 1               # scenario 2: one unit at point 5
F[6, 2] = 1               # scenario 3: one unit at point 7
F[:, 3] = 1               # scenario 4: one unit at every point
F[0, 4] = F[7, 4] = 1     # scenario 5: one unit at each end point (1 and 8)
```

(Counting points from 1, point 2 is the entry with index 1.)

## Tasks

1. **Look at the matrix.** Print `T`. Which entries are nonzero?
2. **Factor it yourself.** Write a function `lu_nopivot(A)` that returns $L$ and $U$ with $A = LU$, using elimination without row exchanges and storing each multiplier in $L$. (For this matrix no exchange is ever needed.) Check that `L @ U` equals `T`, print the diagonal of $U$ and the entries just below the diagonal of $L$, and describe the pattern you see.
3. **Solve all five scenarios with one factorization.** Write `forward_substitute(L, b)` and `back_substitute(U, c)`. Use the same $L$ and $U$ to solve each column of `F`. Compare with `np.linalg.solve(T, F)` and print the largest difference.
4. **Draw the profiles.** Plot the five temperature profiles on one set of axes, with the positions $0, 1, \dots, 9$ on the horizontal axis (include the two end points at temperature $0$). Add a title, axis labels, and a legend. Which scenario gets hottest, and where is the peak of each one-source scenario?
5. **A surprise in the determinant.** Multiply the diagonal of $U$ to get $\det T$ and compare with `np.linalg.det(T)`. Then do the same for rods with 3, 4, and 5 points. What formula does the determinant seem to follow?
6. **Count the saving.** Suppose the rod had $n = 1000$ points stored as an ordinary (dense) matrix, and you needed 1000 scenarios. Using the rough costs from the lesson ($n^3/3$ for a factorization, $n^2$ for each solve), estimate the multiplications if you refactor for every scenario, and if you factor once. What is the ratio?
7. **Say what you saw.** In 4-6 sentences: what pattern did you find in $L$ and $U$, and what does it say about how much work this particular problem needs? Where does the heat peak, and does that match your intuition? Why is factoring once the right choice here?

## What to hand in

A short notebook or script that runs top to bottom, with the plot, plus your written answer to tasks 5 and 7.

## Stretch (optional)

Use 50 points instead of 8 and plot the profile for a single source in the middle. Then compute the full inverse `np.linalg.inv(T)` and look at its columns: each column of the inverse is the profile for a single unit source at one point.
