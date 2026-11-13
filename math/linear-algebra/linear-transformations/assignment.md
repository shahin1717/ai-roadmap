# Assignment: Recover a Transformation

A drawing tool applies a hidden linear transformation to points, and you only see before-and-after pairs. Recover its matrix, check it, chain it with a second transformation, and use the result to move a small shape and then move it back. It takes about 45-60 minutes and needs only NumPy and Matplotlib. Back to the [topic](README.md).

## The setup

The tool was run on six test points. The "after" values were rounded to two decimals, and the measurement carries a little noise (around $\pm 0.03$), so expect a good fit, not a perfect one. All numbers are invented for this exercise.

```python
import numpy as np

before = np.array([(1.0, 0.5), (2.0, -1.0), (-1.0, 1.5), (0.5, 2.0), (-2.0, -0.5), (0.0, 0.0)])
after = np.array([(0.95, 1.17), (2.94, 0.50), (-1.91, 0.67), (-0.42, 2.09), (-2.16, -1.80), (-0.02, 0.00)])

# a small shape to move: a house, as an array with one point per column
house = np.array([(0, 0), (2, 0), (2, 1.5), (1, 2.4), (0, 1.5), (0, 0)], dtype=float).T
```

Each row of `before` is a point $(x, y)$, and the same row of `after` is where the tool sent it.

## Tasks

1. **Recover the matrix.** Assume the tool is a linear map, so `after[i]` is approximately $A\,$`before[i]` for one $2 \times 2$ matrix $A$. Find $A$ by least squares, for example with `np.linalg.lstsq(before, after, rcond=None)` (think about why that call returns $A^\top$). Print $A$, the largest absolute residual, and say in one sentence why the last test point, the origin, supports the assumption of linearity.
2. **Use only two points.** Take the first two test points as the columns of a matrix $X$ and their images as the columns of $Y$. Then $A X = Y$, so $A = Y X^{-1}$. Compute it and compare with task 1. Which estimate would you trust more, and why?
3. **Describe the map.** Compute $\det A$. Does the tool preserve area, and does it flip orientation? Is $A$ invertible? Draw the unit square before and after the map, and mark the images of $\mathbf{e}_1$ and $\mathbf{e}_2$ (they are the columns of $A$).
4. **Chain two maps.** A second tool first reflects left to right (it sends $(x, y)$ to $(-x, y)$) and then stretches $x$ by a factor of 1.5. Write its matrix $B$ as a product of the two simple matrices, in the right order. The tool pipeline "$A$ first, then $B$" has the matrix $C = BA$. Compute $C$ and $AB$, check that they differ, and apply $C$ to `house`. Verify with `np.allclose` that applying $A$ and then $B$ to `house` gives the same result as applying $C$ once. Plot the house before and after.
5. **Undo it.** Compute $C^{-1}$, verify that $C^{-1}C$ is the identity with `np.allclose`, and use $C^{-1}$ to bring the moved house back to the original. In words: which tool must be undone first?
6. **A different basis.** Let $M$ have the first two `before` points as its columns, so they form a new basis. Compute $M^{-1}AM$. Check that its determinant and trace match those of $A$, and say in a sentence why they must.
7. **Report.** In 4-6 sentences, as if to someone who only knows the before-and-after table: what does the recovered map do to the plane, can it be undone, does the order of the two tools matter, and what does the noise in the data do to your estimate of $A$?

## What to hand in

A short notebook or script that runs top to bottom, with the plots, plus your written report for task 7.

## Stretch (optional)

Now suppose the tool also adds a constant offset to every output. These are the "after" values from a second run:

```python
after_shifted = np.array([(1.27, 0.95), (3.20, 0.32), (-1.66, 0.42), (-0.09, 1.90), (-1.83, -2.04), (0.29, -0.22)])
```

Fit a plain linear map as in task 1 and look at the residuals (the origin gives it away). Then add a column of ones to `before` and fit again, so that the last column of the solution is the offset $\mathbf{b}$ in $\mathbf{x} \mapsto A\mathbf{x} + \mathbf{b}$. This is the homogeneous-coordinates trick from the lesson. What offset do you find, and how much did the largest residual shrink?
