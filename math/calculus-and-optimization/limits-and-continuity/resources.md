# Resources: Limits and Continuity

Titles, authors, chapter pointers, and free links only. Items marked *(library)* are good books that are not free to download; borrow or buy them if you can.

## Read

| Source | Where to look | Covers |
|---|---|---|
| Hass, Heil & Weir (based on the work of George B. Thomas, Jr.), *Thomas' Calculus: Early Transcendentals*, 13th ed. *(library)* | Ch. 2 (Limits and Continuity): 2.1 Rates of Change and Tangents to Curves; 2.2 Limit of a Function and Limit Laws (including The Sandwich Theorem, the book's name for the squeeze theorem); 2.3 The Precise Definition of a Limit; 2.4 One-Sided Limits; 2.5 Continuity (including the Intermediate Value Theorem for Continuous Functions); 2.6 Limits Involving Infinity; Asymptotes of Graphs | The whole topic, in the order of this page, with many more worked examples and exercises |
| Same book *(library)* | Ch. 7.4 (Relative Rates of Growth) | Comparing how fast functions such as $\ln x$, powers, and $e^x$ grow |
| Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* ([free PDF](https://mml-book.com)) | Ch. 5.1 (Differentiation of Univariate Functions) | A pointer for the next topic: the derivative as a limit of a difference quotient, with machine learning in mind |
| Aggarwal, *Linear Algebra and Optimization for Machine Learning* *(library)* | Ch. 1.5 (Optimization for Machine Learning), 1.5.1 (The Taylor Expansion for Function Simplification) | A forward pointer: how limits and derivatives are used to approximate loss functions |

## Watch

- MIT OpenCourseWare, *18.01SC Single Variable Calculus*: Session 4 (Limits and Continuity) <https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/1.-differentiation/part-a-definition-and-basic-rules/session-4-limits-and-continuity/> and Session 5 (Discontinuity) <https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/1.-differentiation/part-a-definition-and-basic-rules/session-5-discontinuity/>. The full lecture series is at <https://ocw.mit.edu/courses/18-01-single-variable-calculus-fall-2006/>.
- 3Blue1Brown, *Essence of Calculus*: *Limits and the definition of derivatives* <https://www.3blue1brown.com/lessons/limits>, and *The paradox of the derivative* <https://www.3blue1brown.com/lessons/derivatives>, a companion for the next topic.

## Practice and more explanations

- Paul's Online Math Notes, *Calculus I*, the Limits chapter: <https://tutorial.math.lamar.edu/Classes/CalcI/limitsIntro.aspx>. Its sections include The Limit, One-Sided Limits, Limit Properties, Computing Limits, Infinite Limits, Limits At Infinity, and Continuity, for example <https://tutorial.math.lamar.edu/Classes/CalcI/Continuity.aspx> and <https://tutorial.math.lamar.edu/Classes/CalcI/ComputingLimits.aspx>.
- Khan Academy, the unit on limits and continuity: <https://www.khanacademy.org/math/calculus-1/cs1-limits-and-continuity>. Short videos and practice questions.

## Try

- SymPy documentation: *Calculus* in the introductory tutorial, which covers `limit` <https://docs.sympy.org/latest/tutorials/intro-tutorial/calculus.html>, and the series module reference <https://docs.sympy.org/latest/modules/series/series.html>.
- NumPy documentation: `numpy.log1p` <https://numpy.org/doc/stable/reference/generated/numpy.log1p.html>, `numpy.expm1` <https://numpy.org/doc/stable/reference/generated/numpy.expm1.html>, `numpy.finfo` <https://numpy.org/doc/stable/reference/generated/numpy.finfo.html>, and `numpy.isclose` <https://numpy.org/doc/stable/reference/generated/numpy.isclose.html>.
- SciPy documentation: `scipy.special.logsumexp` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html>, `scipy.special.xlogy` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.xlogy.html>, and `scipy.optimize.bisect`, a library version of the root finder you write in this topic <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.bisect.html>.
- Python documentation, *Floating Point Arithmetic: Issues and Limitations*, on why rounding makes tiny steps misbehave: <https://docs.python.org/3/tutorial/floatingpoint.html>.
- Matplotlib, *Quick start guide*, for the plots in the notebook: <https://matplotlib.org/stable/users/explain/quick_start.html>

## Where this topic is used later

- [Derivatives and Taylor series](../derivatives-and-taylor-series/README.md)
- [Integrals](../integrals/README.md)
- [Multivariable calculus](../multivariable-calculus/README.md)
- [Gradient descent and its variants](../gradient-descent-and-its-variants/README.md)
- [Newton and quasi-Newton methods](../newton-and-quasi-newton-methods/README.md)
- [Neural network basics](../../../deep-learning/neural-network-basics/README.md)
- [Backpropagation](../../../deep-learning/backpropagation/README.md)
