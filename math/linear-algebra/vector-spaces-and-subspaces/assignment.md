# Assignment: The Hidden Redundant Feature

Use the nullspace to discover that one column of a small dataset is redundant, and describe every weight vector that fits the data. It takes about 45-60 minutes and needs only NumPy and SciPy. Back to the [topic](README.md).

## The setup

A fitness app records six days of activity. Each day has four features: minutes walked, minutes cycled, an estimated number of calories burned, and thousands of steps. It also records a recovery score $y$ for the next morning. The numbers are invented for this exercise.

```python
import numpy as np

walk  = np.array([30,  0, 45, 20, 60, 10])
cycle = np.array([ 0, 40, 10, 25,  0, 35])
steps = np.array([ 3,  1,  4,  1,  5,  9])
cal   = 5 * walk + 8 * cycle           # the app's calorie estimate

X = np.column_stack([walk, cycle, cal, steps])      # 6 days x 4 features
y = np.array([189, 403, 382, 373, 375, 437])        # recovery score
```

We want weights $\mathbf{w}$ with $X\mathbf{w} = \mathbf{y}$. (The data were made to fit exactly, so $\mathbf{y}$ is in the column space of $X$.)

## Tasks

1. **Count the pivots.** Print the shape of `X` and its rank with `np.linalg.matrix_rank`. How many free columns does that leave?
2. **Find the redundancy.** Use `scipy.linalg.null_space(X)` to find the nullspace. Scale the basis vector so that its first entry is 1 (divide by its first entry) and read off the combination of columns that adds up to zero. Say in one sentence which feature is redundant and why, using how `cal` was computed.
3. **See the trouble.** Form `G = X.T @ X`. Print its rank and its determinant. Then try `np.linalg.solve(G, X.T @ y)`. Depending on rounding it may raise an error or return absurdly large numbers. Explain why neither result can be trusted.
4. **Find one solution.** Use `w0 = np.linalg.lstsq(X, y, rcond=None)[0]`, which returns one weight vector that fits, and check that `X @ w0` equals `y`. Then build three other weight vectors by adding different multiples of your nullspace vector to `w0`, and check that all of them give exactly the same predictions. Write the complete solution as $\mathbf{w} = \mathbf{w}_0 + t\,\mathbf{n}$ in words.
5. **Remove the redundancy.** Drop the `cal` column and solve the normal equations for the three remaining features, `np.linalg.solve(X3.T @ X3, X3.T @ y)`. What weights do you get? Check that they predict `y`.
6. **Say what it means.** In 4-6 sentences: what does the free variable in this problem stand for in the real world (how much credit goes to the calorie estimate versus walking and cycling)? Why can no amount of data of this kind decide between the infinitely many weight vectors? What could you change about the data collection, or about the model, to make the weights unique?

## What to hand in

A short notebook or script that runs top to bottom, plus your written answers to tasks 2, 3, and 6.

## Stretch (optional)

Add a seventh day where the calorie estimate is not exactly $5\cdot\text{walk} + 8\cdot\text{cycle}$ (say, add 20 calories), and recompute the rank of `X`. What happens to the nullspace, and what does that suggest about real data in which a feature is only *nearly* a combination of others?
