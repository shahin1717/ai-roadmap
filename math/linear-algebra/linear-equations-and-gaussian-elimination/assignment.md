# Assignment: Recover a Curve from Measurements

Use your own elimination code to recover the formula of a curve from a few measurements, then see how fragile the answer can be. It takes about 45-60 minutes and needs only NumPy and Matplotlib. Back to the [topic](README.md).

## The setup

A sensor measures a quantity $y$ at four times $t$. The values below are invented for this exercise, and they were produced by a cubic formula $y = a + bt + ct^2 + dt^3$ whose four coefficients you will recover.

| $t$ | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| $y$ | 2.0 | 4.5 | 8.0 | 15.5 |

```python
t = [0, 1, 2, 3]
y = [2.0, 4.5, 8.0, 15.5]
```

Each measurement gives one equation in the unknowns $a, b, c, d$. Together they form a linear system $V\mathbf{x} = \mathbf{y}$, where each row of $V$ is $(1, t, t^2, t^3)$ for one value of $t$.

## Tasks

1. **Build the system.** Make the $4 \times 4$ matrix `V` with `np.vander(t, increasing=True)` and print it. Write out, in a comment or a sentence, what the second row of `V` means as an equation.
2. **Write the solver.** Write a function `solve_gauss(V, y)` with forward elimination, row exchanges (swap in the row with the largest entry in the column), and back substitution. Use it to find the coefficients. Compare with `np.linalg.solve`, and print the largest difference.
3. **Use the formula.** Predict $y$ at $t = 4$ with your coefficients. Then draw a plot with the four measurements as dots and the fitted cubic as a smooth curve over $0 \le t \le 4$, with a title, axis labels, and a legend.
4. **Make it fragile.** Now take ten times $t = 0, 1, \dots, 9$, generate the exact values from your cubic, and solve for a polynomial of degree 9 (a $10 \times 10$ system). Print the coefficients. Then change one of the ten measurements by $0.01$ and solve again. Compare the largest change in any coefficient with the change you get from the same $0.01$ error in the four-point problem of task 2. (Optionally print `np.linalg.cond(V)` for both matrices: it is a number that measures how sensitive a system is, and larger means more sensitive.)
5. **Say what you saw.** In 4-6 sentences: Did your solver agree with NumPy? What did the tiny measurement error do to the ten-point answer compared with the four-point answer? What does that suggest about fitting a very flexible curve through every measurement exactly?

## What to hand in

A short notebook or script that runs top to bottom, with the plot, plus your written answer to task 5.

## Stretch (optional)

Add noise to the ten values with a seeded generator, `rng = np.random.default_rng(1)` and `rng.normal(0, 0.01, size=10)`, and solve the degree-9 system again. Plot the resulting curve between $t = 0$ and $t = 9$ and describe what it does near the ends. The [least squares](../projections-and-least-squares/README.md) topic shows a better way to fit noisy data.
