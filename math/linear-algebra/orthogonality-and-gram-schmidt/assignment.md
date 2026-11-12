# Assignment: Untangling Correlated Features

Use Gram-Schmidt to turn three overlapping features into three perpendicular ones, and see what is left of each feature after the overlap is removed. It takes about 45-60 minutes and needs only NumPy. Back to the [topic](README.md).

## The setup

A small housing dataset has three features for each of eight houses: the floor area in square meters, the number of rooms, and the garden area in square meters. The numbers are invented for this exercise. Bigger houses tend to have more rooms, so the first two features overlap a lot.

```python
import numpy as np

size   = np.array([ 50,  62,  71,  85,  90, 104, 115, 125], dtype=float)
rooms  = np.array([  2,   3,   3,   4,   4,   5,   5,   6], dtype=float)
garden = np.array([ 10,   0,  25,  15,  40,  20,  35,  30], dtype=float)

X = np.column_stack([size, rooms, garden])      # 8 houses x 3 features
```

Correlation between two features is the cosine of the angle between their **centered** columns (each column minus its mean). So we work with `Xc = X - X.mean(axis=0)`.

## Tasks

1. **Measure the overlap.** Print the correlation matrix `np.corrcoef(X.T)`. Which pair of features overlaps most?
2. **Write Gram-Schmidt.** Write a function `gram_schmidt(A)` that returns `Q` and `R` with `A = Q R` (classical is fine). Apply it to `Xc`, in the order size, rooms, garden. Check with `np.allclose` that `Q @ R` equals `Xc` and that `Q.T @ Q` is the identity.
3. **Compare correlations.** Print the correlation matrix of the columns of `Q`. What do you see, and why is it what it is? Does the first column of `Q` carry the same information as `size`? (Check by comparing it with `Xc[:, 0]` divided by its length.)
4. **Read the factorization.** Print `R`. The diagonal entries are the lengths of what was left after subtracting the overlap with earlier features. Which feature has the smallest leftover relative to its original length? Compute the fraction `R[1, 1]**2 / np.linalg.norm(Xc[:, 1])**2` and say in words what it means for `rooms`.
5. **Does the order matter?** Run Gram-Schmidt again with the columns in the order rooms, size, garden. Compare the two diagonals of `R`. Then check that both `Q` matrices describe the same space: stack them side by side and compute the rank of the result (use `np.linalg.matrix_rank`).
6. **Report.** In 4-6 sentences: what does the second column of `Q` represent in plain words, why did the correlations turn into zeros, and why might orthogonal features be easier to use in a model than the original ones? Mention one thing that is lost by orthogonalizing (hint: the new features are harder to interpret).

## What to hand in

A short notebook or script that runs top to bottom, plus your written report for task 6.

## Stretch (optional)

Compare the condition numbers of `Xc` and `Q` with `np.linalg.cond`. A large condition number means that small errors in the data can cause large errors in a fitted model. Which matrix is the better behaved, and by how much?
