# Assignment: How Many Independent Signals?

A table of sensor readings looks like four signals, but fewer may be doing the real work. You will use rank, bases, and the four subspaces to find out how many independent signals the table holds, and then store it with fewer numbers. It takes about 45-60 minutes and needs only NumPy, SciPy, and SymPy. Back to the [topic](README.md).

## The setup

Four sensors in a room record a value at eight moments. The numbers are invented for this exercise. Each column of `D` is one sensor, and each row is one moment.

```python
import numpy as np

a = np.array([1, 2, 0, 3, 1, 4, 2, 0])          # not given to you in a real project
b = np.array([0, 1, 2, 1, 3, 0, 2, 1])
D = np.column_stack([a, b, a + b, 2 * a - b])    # 8 moments x 4 sensors
```

In a real project you would receive only `D`. (The columns were built from two hidden signals, $a$ and $b$. Pretend you do not know that, and let linear algebra find it.)

## Tasks

1. **Count the signals.** Print the shape of `D` and its rank with `np.linalg.matrix_rank`. How many independent columns does that suggest?
2. **Find a basis.** Use SymPy, `R, pivots = sp.Matrix(D).rref()`, to find the pivot columns. Which sensors form a basis for the column space? Print that $8 \times 2$ matrix `S = D[:, list(pivots)]`.
3. **Read the recipes.** Look at the nonzero rows of $R$. Use them to write each non-basis sensor as a combination of the basis sensors. Then confirm with `scipy.linalg.null_space(D)` that the nullspace has the dimension you expect, and check that each relation you found satisfies `D @ x` equals the zero vector.
4. **The four subspaces.** With $m = 8$, $n = 4$, and $r$ the rank, predict the dimensions of the column space, row space, nullspace, and left nullspace. Check each with `scipy.linalg.orth` and `scipy.linalg.null_space`. Then verify numerically that the row space is perpendicular to the nullspace (the product of their basis matrices should be zero).
5. **Store it smaller.** Let `M` be the first two rows of `R` as a NumPy array. Check that `S @ M` reproduces `D` exactly. Count the numbers stored in `D` and in the pair `S` and `M`. Then count them for a table with 1000 moments and 40 sensors that also has rank 2.
6. **Add realistic noise.** Add noise with `rng = np.random.default_rng(0)` and `D_noisy = D + rng.normal(0, 0.01, D.shape)`. Print the rank with the default tolerance and with `tol=0.1`. Then print `np.linalg.svd(D_noisy, compute_uv=False)`, which lists how strongly the table varies along each direction (the topic on the singular value decomposition explains it). What do you see?
7. **Report.** In 4-6 sentences, tell the person who owns the sensors: how many independent signals the room really has, which two sensors are enough, how the other two are determined, and what you would say about buying a fifth sensor. Mention how you handled noise.

## What to hand in

A short notebook or script that runs top to bottom, plus your written report for task 7.

## Stretch (optional)

Take the hidden signals $a$ and $b$ and add a third independent signal $c$ of your own to a fifth sensor, `a + b + c`. Recompute the rank, the basis, and the four dimensions. Which numbers changed, and which stayed the same?
