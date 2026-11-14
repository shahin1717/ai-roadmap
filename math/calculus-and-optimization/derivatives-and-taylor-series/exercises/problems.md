# Exercises: Derivatives and Taylor Series

Ten problems, easy to harder. Do the first nine on paper (use a calculator, SymPy, or the [notebook](../notebook.ipynb) only to check), then do the last one in code. Worked answers are in [solutions.md](solutions.md). Give each problem a real attempt before you open them.

Back to the [topic](../README.md).

## The derivative and the rules

**1. From the definition.** Use $f'(a) = \lim_{h\to0}\frac{f(a+h)-f(a)}{h}$ (and no rules) to find (a) $f'(2)$ for $f(x) = 3x^2 - x$, and (b) $f'(4)$ for $f(x) = \sqrt{x}$ (hint: multiply top and bottom by the conjugate, as in the limits topic). Then write the tangent line in each case.

**2. The rules.** Differentiate: (a) $2x^5 - 3\sqrt{x} + \dfrac4x$  (b) $x^2\ln x$  (c) $\dfrac{x+1}{x-1}$  (d) $e^x\cos x$.

**3. The chain rule.** Differentiate: (a) $(2x-5)^7$  (b) $\sin(3x^2)$  (c) $e^{-x^2/2}$  (d) $\ln(1+x^2)$  (e) $\sqrt{1 + \sin^2 x}$. For (e), name the three nested functions first.

**4. Sigmoid and tanh.** (a) Use the chain rule to show that $\sigma(x) = \frac{1}{1+e^{-x}}$ has $\sigma'(x) = \sigma(x)(1-\sigma(x))$, then compute $\sigma'(0)$, $\sigma'(3)$, and $\sigma'(-3)$. (b) The function $\tanh x$ equals $2\sigma(2x) - 1$. Use that to show $\tanh'(x) = 1 - \tanh^2 x$. (c) A signal passes backward through $10$ sigmoid layers. What is the largest possible product of the $10$ activation slopes?

**5. Where the derivative fails, and gluing pieces.** (a) At which points is $f(x) = |x^2 - 4|$ not differentiable, and why? (b) Let $g(x) = x^2$ for $x \le 1$ and $g(x) = ax + b$ for $x > 1$. Find $a$ and $b$ so that $g$ is differentiable at $1$ (it must be continuous there and its two one-sided slopes must agree). (c) Leaky ReLU is $x$ for $x > 0$ and $0.1x$ for $x \le 0$. What are its slopes on each side of $0$, and does the derivative exist at $0$?

## Approximation and optimization

**6. Tangent line and approximation error.** (a) Find the tangent line to $y = x^3 - 2x$ at $x = 2$. (b) Linearize $\ln x$ at $a = 1$, and use it to estimate $\ln 1.05$ and $\ln 1.005$. Compute both errors. By what factor did the error shrink when the distance shrank by a factor of $10$, and what does the pattern predict?

**7. Critical points and a one-variable fit.** (a) For $f(x) = x^4 - 8x^2 + 3$, find the critical points, classify them with the first derivative test, confirm with the second derivative test, and find the inflection points. Is there a global minimum? (b) Fit $\hat y = wx$ to the points $(1, 3), (2, 5), (4, 9)$ by minimizing $L(w) = \sum_i (w x_i - y_i)^2$. Find $w$ with calculus and show it is a minimum.

## Taylor polynomials and series

**8. Taylor polynomials.** (a) Write the degree-3 Maclaurin polynomial of $\sin x$ and use it to estimate $\sin 0.3$. Compare the error with the size of the first omitted term, $\frac{0.3^5}{5!}$. (b) Find the degree-2 Taylor polynomial of $\sqrt{x}$ around $a = 4$ and use it to estimate $\sqrt{4.4}$. Compare with the linear estimate.

**9. Series and radius of convergence.** (a) Use the geometric series to write the first four nonzero terms of the Maclaurin series of $\frac{1}{1+x^2}$ and state its radius of convergence. Compute partial sums with $6$ terms at $x = 0.5$ and at $x = 1.5$ and compare with the function. (b) Write the first three nonzero terms of the series of $e^{-x^2}$ by substitution. What is its radius of convergence?

## Code

**10. Checking derivatives in code.** (a) Compare a centered finite difference of the sigmoid with $\sigma(1-\sigma)$ at $x = -3, -1, 0, 2, 5$ using `np.allclose`. Then find, on a grid, the smallest $x > 0$ at which $\sigma'(x)$ falls below $0.01$. (b) For $e^x$, compute the degree-3 Maclaurin error at $x = 0.2$ and at $x = 0.02$, and the ratio of the two errors. What ratio does the $|x|^{n+1}$ rule predict? (c) Starting at $x = 0$, do three Newton jumps $x \leftarrow x - f'(x)/f''(x)$ for $f(x) = e^x - 2x$ and compare with $\ln 2$ using `np.isclose`.
