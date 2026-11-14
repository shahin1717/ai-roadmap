# Exercises: Gradient Descent and its Variants

Ten problems, easy to harder. Do the first nine on paper (use a calculator or the [notebook](../notebook.ipynb) only to check), then do the last one in code. Worked answers are in [solutions.md](solutions.md). Give each problem a real attempt before you open them.

Back to the [topic](../README.md).

## Plain gradient descent

**1. Three steps by hand.** Let $f(x) = x^2 - 6x + 5$ and start at $x_0 = 0$ with $\eta = 0.2$. (a) Compute $x_1, x_2, x_3$. (b) By what factor does the distance to the minimum shrink at each step? (c) How many steps are needed before the distance is below $0.01$?

**2. The learning-rate limit.** The loss $f(x, y) = 3x^2 + xy + y^2$ has the Hessian $`H = \begin{pmatrix}6 & 1\\ 1 & 2\end{pmatrix}`$. (a) Find its eigenvalues. (b) What is the largest learning rate for which gradient descent converges? (c) Which constant learning rate gives the fastest guaranteed convergence, and what is the per-step factor $(\kappa-1)/(\kappa+1)$ there?

**3. A condition-number question.** The loss $f(x, y) = x^2 + 100y^2$ has a long narrow valley. (a) Give $H$, $\kappa$, and the largest safe $\eta$. (b) With the best constant $\eta = 2/(\lambda_{\max} + \lambda_{\min})$, about how many steps does it take to shrink the error by a factor of $1000$? (c) Substitute $y = u/10$. What are $H$ and $\kappa$ in the new coordinates $(x, u)$, and what does that say about feature scaling?

## Stochastic and mini-batch

**4. Cost and noise.** A linear model with $d = 500$ weights is trained on $n = 2\,000\,000$ examples. Computing one example's gradient costs about $2d$ multiply-adds. (a) What does one update cost with the full batch, with $B = 64$, and with $B = 1$? (b) How many updates make one epoch in each case? (c) The per-example gradients have a standard deviation of $40$ in some coordinate. What is the standard deviation of a mini-batch gradient for $B = 64$, and how large must $B$ be to bring it down to $1$?

**5. A momentum step by hand.** Let $f(x, y) = \frac12(x^2 + 10y^2)$, start at $(2, 1)$, with $\eta = 0.05$ and $\beta = 0.8$, using $\mathbf{v} \leftarrow \beta\mathbf{v} + \nabla f$ and $\mathbf{x} \leftarrow \mathbf{x} - \eta\mathbf{v}$. Compute the first three points. Compare the first two with plain gradient descent at the same $\eta$. Which coordinate overshoots, and why?

**6. Adam's first steps by hand.** Take $\eta = 0.01$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, and ignore $\varepsilon$. Start at $(1, 1)$. The gradient is $(4, -0.5)$ at step 1 and $(2, -0.5)$ at step 2. Compute $\mathbf{m}$, $\mathbf{v}$, the bias-corrected $\hat{\mathbf{m}}$ and $\hat{\mathbf{v}}$, and the position after each step. Why is the first step the same size in both coordinates?

**7. A schedule.** You train with $n = 50\,000$ examples and $B = 100$ for $1000$ updates. The schedule is a linear warm-up over the first $W = 100$ updates to $\eta_0 = 0.003$, then a cosine decay to $0$ at update $T = 1000$. (a) Give the learning rate at updates $49$, $100$, $550$, and $1000$. (b) How many epochs is that in total? (c) Why use warm-up, and why decay at the end?

## Judgement

**8. Which optimiser?** Pick a method and say why in one or two sentences. (a) Linear regression, $1000$ rows, $20$ features. (b) Logistic regression on text with $10^6$ sparse features. (c) Fine-tuning a Transformer. (d) A run where the loss reads $2.3, 2.1, 80, \text{nan}$ over the first four steps: what is wrong, and what do you change?

**9. A plateau.** For $f(x) = x^4$ start at $x_0 = 1$ with $\eta = 0.05$. (a) Compute $x_1, x_2, x_3$. (b) The gradient is $4x^3$. Explain why progress slows down as $x \to 0$, even though $x = 0$ is the minimum. (c) If you stop when $\lvert f'(x)\rvert < 10^{-3}$, where does the loop stop, and how far is that from the minimum? What does that say about gradient-norm stopping rules?

## Code

**10. Compare three optimisers.** On $f(x, y) = x^2 + 100y^2$ from problem 3, start at $(5, 1)$ and count the steps until $f < 10^{-8}$ for: (a) gradient descent with $\eta = 2/202$; (b) momentum with $\eta = 0.0165$ and $\beta = 0.67$ (the heavy-ball choice for $\kappa = 100$); (c) Adam with $\eta = 0.1$. Write each update as a small function, and report the three counts. (d) Check the gradient $(2x, 200y)$ against a centered finite difference at $(1, 1)$ with `np.allclose`. (e) Run gradient descent in the rescaled coordinates $(x, u)$ of problem 3(c) from the same starting point, which is $(5, 10)$ there, with $\eta = 0.5$, and count the steps.
