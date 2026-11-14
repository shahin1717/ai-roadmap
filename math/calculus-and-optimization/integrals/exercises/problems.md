# Exercises: Integrals

Ten problems, easy to harder. Do the first nine on paper (use a calculator or the [notebook](../notebook.ipynb) only to check), then do the last one in code. Worked answers are in [solutions.md](solutions.md). Give each problem a real attempt before you open them.

Back to the [topic](../README.md).

## Sums and antiderivatives

**1. A Riemann sum by hand.** Let $f(x) = x^2$ on $[0, 2]$ and use $n = 4$ strips. (a) Compute the left sum and the right sum. (b) Compute the exact area with an antiderivative, and check that it lies between your two sums. (c) How many strips would make the gap between the right and left sums smaller than $0.01$? (The gap is the strip width times the total rise of $f$.)

**2. Antiderivatives.** (a) Find $\displaystyle\int \left(4x^3 - \frac{3}{x} + 2e^x - \sin x\right)dx$ and check your answer by differentiating. (b) Find the function $f$ with $f'(x) = \cos x + 6x$ and $f(0) = 2$.

## The fundamental theorem

**3. Both parts.** (a) Find $\dfrac{d}{dx}\displaystyle\int_0^x e^{-t^2}\,dt$. (b) Find $\dfrac{d}{dx}\displaystyle\int_0^{x^2}\cos t\,dt$ (hint: the upper limit is a function of $x$, so you also need the chain rule). (c) Evaluate $\displaystyle\int_1^4\sqrt{x}\,dx$ and $\displaystyle\int_0^3 (x - 1)\,dx$. (d) For the last integral, also find the total unsigned area between the graph of $x - 1$ and the $x$-axis on $[0, 3]$, and explain the difference.

## Applications

**4. Area between curves.** Find the area enclosed by $y = x^2$ and $y = 8 - x^2$. First find where they cross, then decide which is on top.

**5. Average value and a volume.** (a) Find the average value of $f(x) = 3x^2$ on $[0, 2]$, and the point $c$ in $[0, 2]$ where $f(c)$ equals that average. (b) Spin the region under $y = 1/x$ for $1 \le x \le 4$ around the $x$-axis. Find the volume by discs.

## Techniques

**6. Substitution.** Find (a) $\displaystyle\int 3x^2 e^{x^3}\,dx$, (b) $\displaystyle\int \cos x\,\sin^3 x\,dx$, and (c) $\displaystyle\int_0^1 \frac{x}{(x^2+1)^2}\,dx$, converting the limits in (c).

**7. Integration by parts.** (a) Find $\displaystyle\int x^2 e^x\,dx$ (you will need to apply the method twice). (b) Evaluate $\displaystyle\int_0^\pi x\sin x\,dx$.

**8. Improper integrals.** Decide whether each converges and find the value when it does. (a) $\displaystyle\int_1^\infty x^{-3}\,dx$ (b) $\displaystyle\int_0^\infty 3e^{-3x}\,dx$ (c) $\displaystyle\int_1^\infty \frac{1}{\sqrt{x}}\,dx$ (d) $\displaystyle\int_0^\infty x e^{-x}\,dx$ (use parts, then a limit).

## Probability and code

**9. A density.** (a) Find the constant $c$ that makes $f(x) = c\,(1 - x^2)$ a density on $[0, 1]$ (non-negative with total area $1$). Then find $E[X] = \int_0^1 x f(x)\,dx$ and $P(X \le \tfrac12)$. (b) For the exponential density $f(x) = \lambda e^{-\lambda x}$ on $[0, \infty)$ with $\lambda > 0$, show that the total area is $1$ and find $E[X]$. Then say what $P(X \le a)$ is, as a formula in $a$.

**10. Numerical rules in code.** Write `trapezoid_rule(f, a, b, n)` and `simpson_rule(f, a, b, n)` (Simpson needs $n$ even), both with NumPy. (a) Test them on $\int_0^\pi \sin x\,dx = 2$ for $n = 4, 8, 16, 32$ and print the absolute errors. (b) For each rule, compute the ratio of consecutive errors and check with `np.allclose` (a 10% tolerance is fine) that it is near $4$ for the trapezoid rule and near $16$ for Simpson's rule. (c) Compare with `scipy.integrate.quad` using `np.isclose`, and report the error estimate that `quad` returns.
