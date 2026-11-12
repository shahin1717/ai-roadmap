# Assignment: A Trend and a Season

Fit a trend line to two years of monthly measurements with your own normal equations, look at what the residuals reveal, then add seasonal columns and watch the error shrink. It takes about 45-60 minutes and needs only NumPy and Matplotlib. Back to the [topic](README.md).

## The setup

A small online shop records its monthly number of orders, in hundreds, for 24 months. The numbers are invented for this exercise. They rise over time and also swing up and down with the seasons.

```python
import numpy as np

t = np.arange(1, 25)                    # month number, 1 to 24
y = np.array([109.4, 112.3, 121.8, 123.2, 112.6, 111.1, 111.9, 108.7,
              110.4, 112.0, 124.1, 132.3, 138.7, 148.8, 150.9, 147.8,
              149.6, 142.1, 144.1, 139.5, 139.9, 142.6, 155.2, 159.5])
```

## Tasks

1. **Fit a trend line.** Build the design matrix `A` with a column of ones and the column `t`. Solve the normal equations yourself with `np.linalg.solve(A.T @ A, A.T @ y)`, and compare the result with `np.linalg.lstsq`. Print the intercept, the slope, and the sum of squared residuals. In one sentence, say what the slope means for the shop.
2. **Check the geometry.** Compute the residual vector `r = y - A @ coef`. Verify with `np.allclose` that `A.T @ r` is (nearly) zero, and explain in a sentence why the residuals add up to zero.
3. **Look at the residuals.** Plot the data with the fitted line, and plot the residuals against `t`. Describe the pattern you see in the residuals. What does the pattern tell you that the line is missing?
4. **Add a season.** A yearly cycle can be written with two more columns, $\cos(2\pi t/12)$ and $\sin(2\pi t/12)$. Build a new design matrix with four columns (ones, `t`, cosine, sine), solve the normal equations again, and print the new sum of squared residuals. By what percentage did it shrink?
5. **A surprise in the residuals.** Check that the new residual vector is perpendicular to the *old* two columns as well. Why does that make sense, geometrically?
6. **Hold out the future.** Fit both models using only the first 18 months, then predict months 19 to 24 and compute the root-mean-square error on those six months for each model. Which model predicts the future better, and by how much?
7. **Report.** In 4-6 sentences, as if to the shop owner: how fast are orders growing per month, how big is the seasonal swing (the size of the cycle is $\sqrt{c_3^2 + c_4^2}$, where $c_3$ and $c_4$ are the cosine and sine coefficients), and why you would trust the second model more. Mention one risk of adding more columns.

## What to hand in

A short notebook or script that runs top to bottom, with the plots, plus your written report for task 7.

## Stretch (optional)

Use only the cosine and sine columns for a six-month cycle, $\cos(2\pi t/6)$ and $\sin(2\pi t/6)$, as two extra columns, and see whether the error falls further. Then compare the fit on the first 18 months with the fit on the held-out six. What does the comparison suggest?
