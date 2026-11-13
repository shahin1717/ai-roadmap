# Assignment: The Principal Axes of a Small Dataset

Find the main directions of a small cloud of points three ways: with the characteristic equation, with `eigh`, and with the power method. Then use them to squeeze the data from two numbers per point down to one. It takes about 45-60 minutes and needs only NumPy and Matplotlib. Back to the [topic](README.md).

## The setup

Twelve plants were measured for two quantities, such as the leaf length and the stem length in centimeters. The numbers are invented for this exercise. The two measurements rise and fall together, so the cloud is a tilted ellipse.

```python
import numpy as np

X = np.array([[5.4, 3.2], [3.4, 2.3], [2.9, 2.2], [6.7, 4.1], [3.3, 2.5], [6.4, 3.3],
              [4.1, 2.3], [5.8, 3.6], [1.8, 2.6], [-0.3, 2.0], [1.0, 2.8], [2.8, 3.2]])
```

## Tasks

1. **Center and summarize.** Subtract the mean of each column to get `Xc`, and compute the $2 \times 2$ covariance matrix `C = Xc.T @ Xc / (len(X) - 1)`. Check with `np.allclose` that it matches `np.cov(X.T)`, and that it is symmetric.
2. **The characteristic equation by hand.** For a $2 \times 2$ matrix the characteristic equation is $\lambda^2 - \operatorname{tr}(C)\,\lambda + \det C = 0$. Solve it with the quadratic formula in code (no `eig`), and print both eigenvalues. Why must they be real, and why are both positive?
3. **Eigenvectors by hand.** For each eigenvalue, find an eigenvector as a nullspace vector of `C - lam * np.eye(2)`. For a $2 \times 2$ matrix with rows $(a, b)$ and $(c, d)$ in that matrix, $(b, -a)$ is a nullspace vector if $a$ and $b$ are not both zero. Normalize it to length $1$, and check that the two eigenvectors are orthogonal.
4. **Compare with `eigh`.** Compute `w, V = np.linalg.eigh(C)`. Check that your eigenvalues match `w` and that your eigenvectors agree with the columns of `V` up to a sign (check each with `np.allclose(abs(v1 @ v2), 1)`, where `v1` and `v2` are the two versions). Verify the spectral theorem: `V @ np.diag(w) @ V.T` equals `C`.
5. **Draw it.** Plot the points, the mean, and the two eigenvectors as arrows from the mean, each with length equal to $2\sqrt{\lambda}$. Add a title and labels. Which arrow is longer, and what does its length measure?
6. **The power method.** Write `power_method(A, v0, steps)` and run it on `C` from the start vector `(1, 0)` for 30 steps. Compare the result with the eigenvector for the largest eigenvalue, and the Rayleigh quotient with the largest eigenvalue. Then predict from $|\lambda_2/\lambda_1|$ how many steps are needed to shrink the error by a factor of $10^6$, and check it by running the method and counting.
7. **Compress.** Project each centered point onto the first principal axis: `scores = Xc @ v_top`, which is one number per plant. Compute what fraction of the total variance (the trace of `C`) the variance of `scores` is. Then rebuild the points from the scores alone with `np.outer(scores, v_top) + mean`, and plot the original and rebuilt points together. How much was lost?
8. **Report.** In 4-6 sentences: what do the eigenvectors and eigenvalues of the covariance matrix say about the plants? How much information survives when you keep only one number per plant? In what situation would keeping one number be a bad idea (think of the second eigenvalue being almost as large as the first)?

## What to hand in

A short notebook or script that runs top to bottom, with the plots, plus your written report for task 8.

## Stretch (optional)

Standardize each column (subtract the mean and divide by the standard deviation) before computing the covariance matrix, and redo tasks 1 to 4. Do the eigenvectors change? What does that tell you about how the measurement units affect the principal axes?
