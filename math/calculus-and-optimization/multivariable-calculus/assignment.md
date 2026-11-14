# Assignment: Fit, Check, Classify, Integrate

You will take a two-parameter model, derive the gradient of its loss by hand, verify it numerically, find and classify the critical point with the Hessian, draw the contours, and measure how stretched the valley is. Then you will compute probabilities over regions of an invented 2D density with double integrals. It takes about 60-90 minutes and needs NumPy, SymPy, SciPy, and Matplotlib. Back to the [topic](README.md).

## Part A: a growth curve

The numbers are invented for this exercise. A quantity is measured at seven times, and we model it as $\hat y = a\,e^{bt}$ with two parameters $a$ and $b$.

```python
import numpy as np

t = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3.0])
y = np.array([2.1, 2.6, 3.8, 4.7, 6.9, 8.7, 12.4])
```

The loss is the sum of squared errors, $L(a, b) = \sum_i \left(a\,e^{b t_i} - y_i\right)^2$.

### Tasks

1. **Derive and code the gradient.** Derive $\partial L/\partial a$ and $\partial L/\partial b$ by hand (hint: write $r_i = a\,e^{bt_i} - y_i$ and use the chain rule). Code `loss(p)` and `grad(p)` for $p = (a, b)$, using vectorized NumPy over `t`.
2. **Check it.** Write a centered finite-difference gradient, one coordinate at a time, and compare it with `grad` at $(1.5, 0.5)$ and at $(3, 0.2)$. Print the relative error $\lVert\text{num} - \text{exact}\rVert / (\lVert\text{num}\rVert + \lVert\text{exact}\rVert)$. With $h = 10^{-5}$ you should see about $10^{-9}$ or smaller. If you see $10^{-2}$, look for a bug.
3. **Find the critical point.** Use `scipy.optimize.fsolve(grad, [2.0, 0.5])` to solve $\nabla L = \mathbf{0}$. Print the solution, the gradient there (it should be near zero, so compare with `np.allclose`), and the loss.
4. **Classify it.** Build the Hessian of $L$ with SymPy (`sp.hessian`) and evaluate it at the critical point. Print its eigenvalues with `np.linalg.eigvalsh` and say whether the point is a minimum, a maximum, or a saddle. Report the condition number.
5. **Draw the contours.** Plot at least 20 contour lines of $L$ over $a\in[1, 3]$, $b\in[0.3, 0.8]$, mark the critical point, and draw the two Hessian eigenvector directions through it as arrows. Check that the labels do not overlap. Which direction is the flat one?
6. **Change the parameters.** Replace $t$ by $t - \bar t$ (subtract the mean time) and refit the model $\hat y = a\,e^{b(t - \bar t)}$. Report the new critical point, the new Hessian condition number, and what happened to $b$ and to the loss. Expect the condition number to drop by a large factor while the loss stays the same.
7. **Explain.** In 4-6 sentences: why is the valley long and narrow in the original parameters, what does the condition number tell you about it, and why does shifting $t$ help? (Think about how a change in $b$ moves the fitted values at large $t$, and how $a$ can compensate.)

## Part B: a probability over a region

A pair of measurements $(X, Y)$ in the unit square has the density $p(x, y) = c\,(x + 2y)$ for $0\le x\le1$, $0\le y\le1$, and $0$ elsewhere.

1. **Normalise.** Find $c$ so that $\iint p\,dA = 1$, by hand and then with SymPy.
2. **A triangular region.** Compute $P(X + Y \le 1)$ with an iterated integral whose inner limits depend on the outer variable. Do it in both orders with SymPy, and check one of them with `scipy.integrate.dblquad`. (The answer is a simple fraction.)
3. **Another one.** Compute $P(Y < X)$ the same way.
4. **Simulate.** Draw $N = 400000$ points $(x, y)$ uniformly in the square, and keep each with probability $p(x, y)/2$ (this is called rejection sampling, and it works because $p \le 2$ on the square). Using a seeded generator, estimate both probabilities from the kept points and compare with your exact answers with `np.isclose` and a tolerance of about $0.005$.

## What to hand in

A short notebook or script that runs top to bottom, with the contour plot, plus your written explanations for tasks A5 to A7 and a line or two for each task in Part B.

## Stretch (optional)

Try a few other starting guesses for `fsolve` in task A3 (for example `[0.1, 1.5]`) and check whether each one reaches the same critical point. Plot the gradient field of $L$ (arrows) over the contours and look at where the gradient is small but not zero. Then say, in two sentences, why a root finder on $\nabla L$ is not the same thing as walking downhill (the topic of the [next lesson](../gradient-descent-and-its-variants/README.md)).
