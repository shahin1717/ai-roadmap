# Exercises: Limits and Continuity

Ten problems, easy to harder. Do the first nine on paper (use a calculator or the [notebook](../notebook.ipynb) only to check), then do the last one in code. Worked answers are in [solutions.md](solutions.md). Give each problem a real attempt before you open them.

Back to the [topic](../README.md).

## Seeing a limit

**1. A table of values.** Evaluate $f(x) = \dfrac{e^x - 1}{x}$ at $x = -0.1, -0.01, -0.001, 0.001, 0.01, 0.1$. What number does $f(x)$ seem to be heading toward as $x \to 0$? Is $f(0)$ defined?

**2. Sketch and decide.** Sketch a graph with all of these properties: $\lim_{x\to1} f(x) = 2$ but $f(1) = 4$; $\lim_{x\to3^-} f(x) = 1$, $\lim_{x\to3^+} f(x) = 3$, and $f(3) = 1$; and $f$ is continuous everywhere else. At $x = 1$ and at $x = 3$, say whether $f$ is continuous, and if not, which type of discontinuity it has. Which of the two could you repair by changing the value of $f$ at that single point?

## Computing limits

**3. Algebra.** Find each limit, saying which rewrite you used.

(a) $\displaystyle\lim_{x\to4}\frac{x^2-16}{x-4}$  (b) $\displaystyle\lim_{x\to0}\frac{\sqrt{x+25}-5}{x}$  (c) $\displaystyle\lim_{h\to0}\frac{(3+h)^2-9}{h}$  (d) $\displaystyle\lim_{x\to1}\frac{\frac1x-1}{x-1}$

**4. One-sided limits.** (a) Find $\lim_{x\to2^-}\dfrac{|x-2|}{x-2}$ and $\lim_{x\to2^+}\dfrac{|x-2|}{x-2}$. Does the two-sided limit exist? (b) For $g(x) = \dfrac{x^2-4}{|x-2|}$, find both one-sided limits at $2$ and decide whether the limit exists.

**5. The squeeze theorem.** Use the squeeze theorem for both: (a) $\displaystyle\lim_{x\to0} x^2\sin\frac1x$; (b) $\displaystyle\lim_{x\to\infty}\frac{\sin x}{x}$. Write down the two walls in each case.

**6. A tolerance game.** For $f(x) = 4x - 5$, $a = 2$, $L = 3$: find a $\delta$ (in terms of $\varepsilon$) that answers any $\varepsilon > 0$, and give the specific $\delta$ for $\varepsilon = 0.1$ and for $\varepsilon = 0.001$. Then explain why a $\delta$ of $\varepsilon/2$ would **not** be a safe answer for every $\varepsilon$.

## Continuity

**7. Fixing a function.** (a) Let $f(x) = \dfrac{x^2 - x - 6}{x - 3}$ for $x \neq 3$. What value of $f(3)$ makes $f$ continuous at $3$? (b) Let $g(x) = ax + 1$ for $x < 2$ and $g(x) = x^2 - a$ for $x \ge 2$. For which value of $a$ is $g$ continuous at $2$?

**8. Trapping a root.** Show that $x^5 + x - 1 = 0$ has a solution between $0$ and $1$. Then do three steps of bisection by hand (write the bracket after each step). How many bisection steps would you need, starting from $[0, 1]$, to pin the root down to within $10^{-8}$?

## Infinity

**9. Asymptotes and growth.** (a) Find $\displaystyle\lim_{x\to\infty}\frac{4x^2-3x+1}{2x^2+5}$ and the horizontal asymptote. (b) Find $\displaystyle\lim_{x\to\infty}\left(\sqrt{x^2+x}-x\right)$ (hint: multiply by the conjugate). (c) Find the vertical asymptote of $\dfrac{x^2+1}{x-1}$ and the two one-sided limits there. (d) Compute $x^4$ and $2^x$ at $x = 10, 15, 16, 17, 30$. Which one wins for large $x$, and where does the lead change hands?

## Code

**10. Bisection from scratch.** Write `bisect(func, a, b, tol)` that returns the approximate root and the number of steps. (a) Use it to solve $\cos x = x$ on $[0, 1]$ with tolerances $10^{-3}$, $10^{-6}$, and $10^{-10}$, and compare the number of steps with $\log_2\big((b-a)/\text{tol}\big)$. (b) Check your root against `scipy.optimize.brentq` using `np.isclose`. (c) Make your function raise a clear error when the starting values do not have opposite signs, and test it on $x^2 + 1$.
