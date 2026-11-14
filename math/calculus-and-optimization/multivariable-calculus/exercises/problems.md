# Exercises: Multivariable Calculus

Ten problems, easy to harder. Do the first nine on paper (use a calculator or the [notebook](../notebook.ipynb) only to check), then do the last one in code. Worked answers are in [solutions.md](solutions.md). Give each problem a real attempt before you open them.

Back to the [topic](../README.md).

## Partial derivatives

**1. Partials by hand.** Find all first partial derivatives.

(a) $f(x, y) = 3x^2y - y^3 + 5x$, and evaluate both partials at $(1, 2)$  (b) $g(x, y) = x\,e^{2y}$  (c) $h(x, y, z) = xyz + z^2$

(d) For $f(x, y) = xy^2$, use the limit definition $\displaystyle\lim_{h\to0}\frac{f(1, 2+h) - f(1, 2)}{h}$ to find $f_y(1, 2)$.

**2. Mixed partials.** For $g(x, y) = x\,e^{y^2} + \cos(xy)$ compute $g_{xy}$ and $g_{yx}$ and show they agree. Then check at the point $(0.5, 1.2)$ with finite differences, taking the two differences in both orders.

## The chain rule

**3. Along a curve, and through two inner functions.**

(a) Let $f(x, y) = x^2 + 3xy$ with $x = t^2$ and $y = t + 1$. Find $df/dt$ at $t = 1$ with the chain rule, and check it by substituting first.

(b) Let $f(u, v) = u^2v$ with $u = x + y$ and $v = x - y$. Find $\partial f/\partial x$ and $\partial f/\partial y$ at $(1, 2)$, and check by expanding $f$ in terms of $x$ and $y$.

## Gradient, directional derivative, Jacobian

**4. Gradient and directional derivative.** Let $f(x, y) = x^2y + y^2$ at the point $(1, 2)$. (a) Find $\nabla f$. (b) Find the slope in the direction of $(3, 4)$. (c) In which direction does $f$ increase fastest, and at what rate? (d) Give a direction in which $f$ does not change at that point.

**5. A Jacobian.** Let $F(x, y) = (e^x\cos y,\; e^x\sin y)$. (a) Find $J_F$ and its determinant, and evaluate both at $(0, \pi/2)$. (b) Let $G(u, v) = u^2 + v^2$. Use $J_G\,J_F$ to find the gradient of $G(F(x, y))$, and compare with differentiating $G(F(x, y))$ directly after simplifying.

## Hessian and approximation

**6. Classify the critical points.** For $f(x, y) = x^3 + y^3 - 3xy$, find every critical point, build the Hessian, and classify each point with its eigenvalues. For a minimum, give the condition number of the Hessian.

**7. Linear and quadratic approximations.** Let $f(x, y) = \sqrt{x^2 + y^2}$. Use the linear approximation at $(3, 4)$ to estimate $f(3.04, 3.97)$. Then add the Hessian term and compare both estimates with the true value. (Hint: the step is perpendicular to the gradient, so the linear estimate cannot change.)

## Double integrals

**8. Iterated integrals.** (a) Compute $\iint f\,dA$ for $f = 3x^2 + 2xy$ over the rectangle $[0, 1]\times[0, 2]$. (b) Let $R$ be the region between the curves $y = x^2$ and $y = x$ for $0 \le x \le 1$. Compute $\iint_R xy\,dA$ by integrating in $y$ first, then swap the order and check.

**9. Polar coordinates.** (a) Compute $\iint e^{-(x^2+y^2)}\,dA$ over the disc $x^2 + y^2 \le 4$. (b) What is the average value of $x^2 + y^2$ over the unit disc? (c) Compute $\iint xy\,dA$ over the part of the unit disc in the first quadrant.

## Code

**10. Numerical gradient and Hessian.** (a) Write `num_grad(func, p, h)` with centered differences, one coordinate at a time. (b) Write `num_hessian(func, p, h)` by applying `num_grad` to each coordinate of the gradient, and symmetrise the result. (c) Test both on $f(x, y) = x^3 + y^3 - 3xy$ at its two critical points from problem 6: check that the gradient is near zero, that the Hessian matches SymPy with `np.allclose`, and classify each point with `np.linalg.eigvalsh`. (d) Check `num_grad` on the three-variable function $x^2 + 2y^2 + 3z^2 + xy$ at $(1, 1, 1)$, whose exact gradient is $(3, 5, 6)$.
