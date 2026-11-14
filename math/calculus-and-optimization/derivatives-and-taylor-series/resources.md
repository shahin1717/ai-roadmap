# Resources: Derivatives and Taylor Series

Titles, authors, chapter pointers, and free links only. Items marked *(library)* are good books that are not free to download; borrow or buy them if you can.

## Read

| Source | Where to look | Covers |
|---|---|---|
| Hass, Heil & Weir (based on the work of George B. Thomas, Jr.), *Thomas' Calculus: Early Transcendentals*, 13th ed. *(library)* | Ch. 3 (Derivatives): 3.1 Tangents and the Derivative at a Point; 3.2 The Derivative as a Function; 3.3 Differentiation Rules; 3.4 The Derivative as a Rate of Change; 3.5 Derivatives of Trigonometric Functions; 3.6 The Chain Rule; 3.7 Implicit Differentiation; 3.8 Derivatives of Inverse Functions and Logarithms; 3.10 Related Rates; 3.11 Linearization and Differentials | Sections 1 to 5 of this topic, with many more worked examples and exercises |
| Same book *(library)* | Ch. 4 (Applications of Derivatives): 4.1 Extreme Values of Functions; 4.3 Monotonic Functions and the First Derivative Test; 4.4 Concavity and Curve Sketching; 4.6 Applied Optimization; 4.7 Newton's Method | Section 6 of this topic, and a forward pointer to Newton's method |
| Same book *(library)* | Ch. 10 (Infinite Sequences and Series): 10.7 Power Series; 10.8 Taylor and Maclaurin Series; 10.9 Convergence of Taylor Series | Sections 7 and 8 of this topic: Taylor polynomials, the remainder, radius of convergence |
| Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* ([free PDF](https://mml-book.com)) | Ch. 5 (Vector Calculus): 5.1 Differentiation of Univariate Functions; 5.6 Backpropagation and Automatic Differentiation; 5.7 Higher-Order Derivatives; 5.8 Linearization and Multivariate Taylor Series | The same ideas with machine learning in mind, and the bridge to several variables |
| Aggarwal, *Linear Algebra and Optimization for Machine Learning* *(library)* | Ch. 1.5 (Optimization for Machine Learning), 1.5.1 (The Taylor Expansion for Function Simplification) and 1.5.3 (Optimization in Computational Graphs); Ch. 4.2.1 (Univariate Optimization); Ch. 4.4.1 (Checking Gradient Correctness with Finite Differences) | Taylor models of a loss, one-variable optimization, and checking derivatives numerically |
| Baydin, Pearlmutter, Radul & Siskind, *Automatic differentiation in machine learning: a survey* ([arXiv](https://arxiv.org/abs/1502.05767), [PDF](https://arxiv.org/pdf/1502.05767)) | The overview of symbolic, numeric, and automatic differentiation, and the forward and reverse modes | How symbolic, numeric, and automatic differentiation differ, and why reverse mode is used for training |

## Watch

- MIT OpenCourseWare, *18.01SC Single Variable Calculus*: Part A of Unit 1, [Definition and Basic Rules](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/1.-differentiation/part-a-definition-and-basic-rules/), including [Session 1, Introduction to Derivatives](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/1.-differentiation/part-a-definition-and-basic-rules/session-1-introduction-to-derivatives/), [Session 11, Chain Rule](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/1.-differentiation/part-a-definition-and-basic-rules/session-11-chain-rule/), and [Session 12, Higher Derivatives](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/1.-differentiation/part-a-definition-and-basic-rules/session-12-higher-derivatives/). Unit 2, [Approximation and Curve Sketching](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/unit-2-applications-of-differentiation/part-a-approximation-and-curve-sketching/) covers linear and quadratic approximation (Sessions 23 to 26), and Unit 5, [Taylor Series](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/unit-5-exploring-the-infinite/part-b-taylor-series/) covers Taylor series. The full lecture series is at <https://ocw.mit.edu/courses/18-01-single-variable-calculus-fall-2006/>.
- 3Blue1Brown, *Essence of Calculus*: [The paradox of the derivative](https://www.3blue1brown.com/lessons/derivatives), [Visualizing the chain rule and product rule](https://www.3blue1brown.com/lessons/chain-rule-and-product-rule), [Implicit differentiation, what's going on here?](https://www.3blue1brown.com/lessons/implicit-differentiation), and [Taylor series](https://www.3blue1brown.com/lessons/taylor-series). The whole series is at <https://www.3blue1brown.com/topics/calculus>.

## Practice and more explanations

- Paul's Online Math Notes, *Calculus I*, the Derivatives chapter: <https://tutorial.math.lamar.edu/Classes/CalcI/DerivativeIntro.aspx>. Useful sections include [The Definition of the Derivative](https://tutorial.math.lamar.edu/Classes/CalcI/DefnOfDerivative.aspx), [Differentiation Formulas](https://tutorial.math.lamar.edu/Classes/CalcI/DiffFormulas.aspx), and [Chain Rule](https://tutorial.math.lamar.edu/Classes/CalcI/ChainRule.aspx). The Applications chapter has [Minimum and Maximum Values](https://tutorial.math.lamar.edu/Classes/CalcI/MinMaxValues.aspx), [The Shape of a Graph, Part I](https://tutorial.math.lamar.edu/Classes/CalcI/ShapeofGraphPtI.aspx), [Part II](https://tutorial.math.lamar.edu/Classes/CalcI/ShapeofGraphPtII.aspx), and [Linear Approximations](https://tutorial.math.lamar.edu/Classes/CalcI/LinearApproximations.aspx). For series, see *Calculus II*: [Power Series](https://tutorial.math.lamar.edu/Classes/CalcII/PowerSeries.aspx) and [Taylor Series](https://tutorial.math.lamar.edu/Classes/CalcII/TaylorSeries.aspx).
- Khan Academy, the differential calculus unit: <https://www.khanacademy.org/math/differential-calculus>. Short videos and practice questions on derivatives, the chain rule, and applications.

## Try

- SymPy documentation: *Calculus* in the introductory tutorial, which covers `diff`, `limit`, and `series` <https://docs.sympy.org/latest/tutorials/intro-tutorial/calculus.html>, and the series module reference <https://docs.sympy.org/latest/modules/series/series.html>. `lambdify` turns a SymPy expression into a fast NumPy function <https://docs.sympy.org/latest/modules/utilities/lambdify.html>.
- NumPy documentation: `numpy.isclose` <https://numpy.org/doc/stable/reference/generated/numpy.isclose.html>, `numpy.allclose` <https://numpy.org/doc/stable/reference/generated/numpy.allclose.html>, `numpy.polyfit` (used to measure slopes on log-log plots) <https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html>, and `numpy.gradient`, a finite-difference derivative of sampled data <https://numpy.org/doc/stable/reference/generated/numpy.gradient.html>.
- Python documentation, *Data model*, the special methods such as `__add__` and `__mul__` that the dual-number class uses: <https://docs.python.org/3/reference/datamodel.html>.
- Matplotlib, *Quick start guide*, for the plots in the notebook: <https://matplotlib.org/stable/users/explain/quick_start.html>

## Where this topic is used later

- [Integrals](../integrals/README.md)
- [Multivariable calculus](../multivariable-calculus/README.md)
- [Gradient descent and its variants](../gradient-descent-and-its-variants/README.md)
- [Newton and quasi-Newton methods](../newton-and-quasi-newton-methods/README.md)
- [Neural network basics](../../../deep-learning/neural-network-basics/README.md)
- [Backpropagation](../../../deep-learning/backpropagation/README.md)
