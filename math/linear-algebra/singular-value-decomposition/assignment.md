# Assignment: Compress a Picture with the SVD

Squeeze a small synthetic picture through the SVD at several ranks, measure what you lose, and decide the smallest rank that keeps $95\%$ of the energy. It takes about 45-60 minutes and needs only NumPy and Matplotlib. Back to the [topic](README.md).

## The setup

The function below draws an invented $64 \times 96$ "city at sunset": a sky that is bright at the top, a sun, a row of dark buildings of random width and height, and rows of lit windows, plus a little noise. Pixel values are brightness numbers, roughly between $0.1$ and $1.2$.

```python
import numpy as np
import matplotlib.pyplot as plt


def make_scene(m=64, n=96, seed=3):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:m, 0:n]
    r, c = yy / (m - 1), xx / (n - 1)
    img = 0.9 - 0.5 * r                                                            # sky
    img += 0.4 * np.exp(-((r - 0.25) ** 2 + (c - 0.75) ** 2) / (2 * 0.07 ** 2))    # a sun
    edges = np.sort(rng.choice(np.arange(6, n - 6), size=7, replace=False))
    edges = np.concatenate([[0], edges, [n]])
    for left, right in zip(edges[:-1], edges[1:]):                                 # buildings
        top = rng.integers(int(0.35 * m), int(0.75 * m))
        img[top:, left:right] = 0.15 + 0.05 * rng.random()
        img[top + 3:m - 3:6, left + 2:right - 2:4] = 0.7                           # lit windows
    img += 0.02 * rng.normal(size=(m, n))                                          # a little noise
    return img


img = make_scene()
```

## Tasks

1. **Look and center.** Show `img` with `plt.imshow(img, cmap="gray")`. Subtract the overall mean brightness to get `A = img - img.mean()`, and keep the mean aside. (Without this step the "average brightness" would dominate the energy and the first layer would swallow almost everything.) Print the shape of `A`.
2. **Factor.** Compute the thin SVD of `A`. Check with `np.allclose` that $U\Sigma V^\top$ rebuilds `A`, that `U` and `Vt` have orthonormal columns and rows, and that the singular values are sorted. Plot the singular values on a log scale (a scree plot) and describe the shape you see.
3. **Choose the rank.** Compute the cumulative energy $\sum_{i \le k}\sigma_i^2 / \sum_i\sigma_i^2$ for every $k$ and plot it. Find the smallest $k$ that keeps at least $95\%$ of the energy, and the smallest that keeps $99\%$. Print both.
4. **See it.** For $k = 1, 2, 4, 8, 16, 32$, build the rank-$k$ approximation $A_k = U_k\Sigma_kV_k^\top$, add the mean back, and show all six next to the original in a grid. At what $k$ do the buildings, the sun, and the windows each become visible? Does the $95\%$ rank from task 3 look good enough to you? Is the $99\%$ rank?
5. **Check Eckart-Young.** For each of those $k$, compute the 2-norm error $\|A - A_k\|_2$ (`np.linalg.norm(A - Ak, 2)`) and compare it with $\sigma_{k+1}$. Also check that the relative Frobenius error $\|A - A_k\|_F/\|A\|_F$ equals $\sqrt{1 - \text{energy kept}}$. Print a small table with the three quantities side by side.
6. **Count the savings.** A rank-$k$ approximation stores $k(m + n + 1)$ numbers. For each $k$ above, print the number stored and the percentage of $m \cdot n$. Compute the break-even rank $mn/(m+n+1)$ and say what happens to the "compression" beyond it.
7. **Report.** In 4-6 sentences: which rank would you choose for this picture and why? What is lost first when you lower $k$ (large smooth regions, or thin sharp details)? Why does a picture with many small, unrelated details need a larger $k$ than one made of a few smooth shapes?

## What to hand in

A short notebook or script that runs top to bottom, with the plots and the tables, plus your written report for task 7.

## Stretch (optional)

Add stronger noise, `img_noisy = img + 0.1 * np.random.default_rng(0).normal(size=img.shape)`, and redo tasks 2 and 3. How does the scree plot change, and what happens to the rank needed for $95\%$? Then keep a small $k$ and compare the rank-$k$ approximation of the noisy image with the original clean `img`: is it closer to the clean picture than the noisy one is? Measure it with the root-mean-square difference.
