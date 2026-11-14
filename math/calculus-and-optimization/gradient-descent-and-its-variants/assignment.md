# Assignment: Four Optimisers, One Loss

You will fit a three-parameter model to invented data with four optimisers that you write yourself (gradient descent, mini-batch SGD, momentum, and Adam), check your gradient, measure how scaling the features changes the safe learning rate, and compare the optimisers across learning rates and batch sizes. Then you report the best settings and explain the result in a few sentences. It takes about 60-90 minutes and needs only NumPy and Matplotlib. Back to the [topic](README.md).

## The data

The numbers are invented for this exercise. A price (in units of 100 000) depends on the size of a flat and its number of rooms. Rooms grow with size, so the two features are strongly correlated.

```python
import numpy as np

rng = np.random.default_rng(42)
n = 200
size = rng.uniform(40, 200, n)                        # square metres
rooms = size / 35 + rng.normal(0, 0.4, n)             # correlated with size
price = (1.5 * size + 10 * rooms + 30 + rng.normal(0, 12, n)) / 100
X_raw = np.column_stack([size, rooms, np.ones(n)])    # the last column is the intercept
```

The model is $\hat y = w_1\,\text{size} + w_2\,\text{rooms} + b$ with the loss $L(\mathbf{w}) = \frac1{2n}\sum_i(\hat y_i - y_i)^2$, so there are three parameters $(w_1, w_2, b)$.

## Tasks

1. **Gradient.** Write `loss(w, X)` and `grad(w, X, y)` (vectorised). Check `grad` against a centered finite difference at two points with `np.allclose` and report the relative error. Break the gradient on purpose (drop the $1/n$) and show that the check catches it.
2. **Conditioning.** Form the Hessian $X^\top X/n$ for `X_raw`. Report its eigenvalues, the condition number $\kappa$, and the largest safe learning rate $2/\lambda_{\max}$. Run gradient descent from $\mathbf{w} = \mathbf{0}$ with a rate just below and just above that limit and show what each does to the loss over $200$ steps.
3. **Scaling.** Standardise the two features (subtract the mean, divide by the standard deviation, keep the intercept column of ones) to get `X_std`. Report the new eigenvalues, $\kappa$, and safe rate. Get the exact answer with `np.linalg.lstsq` and call its loss `best`. From now on measure the **excess loss** $L(\mathbf{w}) - \text{best}$.
4. **Four optimisers from scratch.** Write gradient descent, mini-batch SGD (with a shuffle each epoch), momentum ($\mathbf{v} \leftarrow \beta\mathbf{v} + \mathbf{g}$, $\mathbf{w} \leftarrow \mathbf{w} - \eta\mathbf{v}$, with $\beta = 0.9$), and Adam (with bias correction and the default $\beta_1$, $\beta_2$, $\varepsilon$). Check your Adam against a hand computation of its first step: every coordinate should move by about $\eta$.
5. **Learning-rate sweep.** On `X_std`, for each of gradient descent, momentum, and Adam run a grid of learning rates, for example $0.01, 0.03, 0.1, 0.3, 1$ (extend the grid where a method still improves). Count the updates until the excess loss drops below $10^{-4}$, capped at $300$ epochs. Mark runs that diverge or never get there.
6. **Batch-size sweep.** Run mini-batch SGD with batch sizes $1$, $10$, $50$, and $200$ (the full batch) and a few learning rates each. Report epochs and updates to reach the same tolerance. Remember that one epoch is $n/B$ updates, so compare both counts.
7. **Plot.** Draw the excess loss against the number of epochs (log scale) for the best run of each method on one figure, and the contour of the loss over $(w_1, w_2)$ for `X_std` with the path of gradient descent and of momentum drawn on top.
8. **Report.** In 5-8 sentences: which settings were best for each method, why the safe learning rate for `X_raw` is so much smaller than for `X_std`, what the condition number of `X_std` says about how much momentum or Adam can help here, and what the batch-size sweep tells you about the noise of SGD and the number of epochs it needs.

## What to hand in

A short notebook or script that runs top to bottom with the figures, your tables of counts, and your written report. Quote patterns more than exact numbers, because the counts depend on your seeds and your grid.

## Stretch (optional)

Add a cosine learning-rate schedule to mini-batch SGD with $B = 1$ and show that it reaches a lower excess loss than the best constant rate. Then add a fifth optimiser, RMSProp, and say in one sentence how its first step differs from Adam's and why.
