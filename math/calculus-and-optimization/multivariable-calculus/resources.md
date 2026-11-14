# Resources: Multivariable Calculus

Titles, authors, chapter pointers, and free links only. Items marked *(library)* are good books that are not free to download; borrow or buy them if you can.

## Read

| Source | Where to look | Covers |
|---|---|---|
| Hass, Heil & Weir (based on the work of George B. Thomas, Jr.), *Thomas' Calculus: Early Transcendentals*, 13th ed. *(library)* | Ch. 14 (Partial Derivatives): 14.1 Functions of Several Variables; 14.3 Partial Derivatives; 14.4 The Chain Rule; 14.5 Directional Derivatives and Gradient Vectors; 14.7 Extreme Values and Saddle Points; 14.9 Taylor's Formula for Two Variables | Sections 1 to 5 of this page, with many more worked examples and exercises |
| Same book *(library)* | Ch. 15 (Multiple Integrals): 15.1 Double and Iterated Integrals over Rectangles; 15.2 Double Integrals over General Regions; 15.4 Double Integrals in Polar Form; 15.8 Substitutions in Multiple Integrals | Sections 6 and 7: Riemann sums, Fubini's theorem, regions, polar coordinates, and the Jacobian in a change of variables |
| Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* ([free PDF](https://mml-book.com)) | Ch. 5 (Vector Calculus): 5.2 Partial Differentiation and Gradients; 5.3 Gradients of Vector-Valued Functions; 5.4 Gradients of Matrices; 5.5 Useful Identities for Computing Gradients; 5.6 Backpropagation and Automatic Differentiation; 5.7 Higher-Order Derivatives; 5.8 Linearization and Multivariate Taylor Series | The gradient, the Jacobian, the Hessian, and the chain rule with machine learning in mind |
| Same book ([free PDF](https://mml-book.com)) | Ch. 6.7 (Change of Variables/Inverse Transform) | How a density changes under a change of variables, the idea behind normalising flows |
| Aggarwal, *Linear Algebra and Optimization for Machine Learning* *(library)* | Ch. 4 (Optimization Basics: A Machine Learning View): 4.2 The Basics of Optimization (4.2.2 Bivariate Optimization, 4.2.3 Multivariate Optimization); 4.3 Convex Objective Functions; 4.4.1 Checking Gradient Correctness with Finite Differences; 4.6 Computing Derivatives with Respect to Vectors (including 4.6.3 The Chain Rule of Calculus for Vectored Derivatives); 4.7 Linear Regression: Optimization with Numerical Targets | Gradients and Hessians of loss functions, gradient checking, matrix calculus, and the least-squares gradient |
| Parr & Howard, *The Matrix Calculus You Need For Deep Learning* ([arXiv:1802.01528](https://arxiv.org/abs/1802.01528)) | The whole paper | Jacobians and the chain rule for vector functions, step by step, aimed at deep learning |
| Petersen & Pedersen, *The Matrix Cookbook* ([PDF](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)) | The chapter on derivatives | A reference list of gradient identities to look up, not to read through |

## Watch

- MIT OpenCourseWare, *18.02SC Multivariable Calculus*: the partial derivatives unit, including functions of two variables and tangent approximation <https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/pages/2.-partial-derivatives/part-a-functions-of-two-variables-tangent-approximation-and-optimization/>, and the chain rule, gradient, and directional derivatives <https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/pages/2.-partial-derivatives/part-b-chain-rule-gradient-and-directional-derivatives/>. For double integrals: <https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/pages/3.-double-integrals-and-line-integrals-in-the-plane/part-a-double-integrals/>. The full lecture series is at <https://ocw.mit.edu/courses/18-02-multivariable-calculus-fall-2007/>.
- 3Blue1Brown, *Backpropagation calculus* <https://www.3blue1brown.com/lessons/backpropagation-calculus>: the chain rule applied to a tiny network, a companion to the machine learning section.
- Khan Academy, the multivariable calculus course: <https://www.khanacademy.org/math/multivariable-calculus>. Short videos and practice on partial derivatives, gradients, and multiple integrals (the site did not load for our link check, so look for the units by name).

## Practice and more explanations

- Paul's Online Math Notes, *Calculus III*: the Partial Derivatives chapter <https://tutorial.math.lamar.edu/Classes/CalcIII/PartialDerivatives.aspx>, with sections on the chain rule <https://tutorial.math.lamar.edu/Classes/CalcIII/ChainRule.aspx>, directional derivatives <https://tutorial.math.lamar.edu/Classes/CalcIII/DirectionalDeriv.aspx>, and relative minimums and maximums <https://tutorial.math.lamar.edu/Classes/CalcIII/RelativeExtrema.aspx>. For integrals: double integrals <https://tutorial.math.lamar.edu/Classes/CalcIII/DoubleIntegrals.aspx> and double integrals in polar coordinates <https://tutorial.math.lamar.edu/Classes/CalcIII/DIPolarCoords.aspx>.

## Try

- SymPy documentation: the *Calculus* page of the introductory tutorial for `diff` and `integrate` <https://docs.sympy.org/latest/tutorials/intro-tutorial/calculus.html>, and the matrices reference for `jacobian` and `hessian` <https://docs.sympy.org/latest/modules/matrices/matrices.html>.
- NumPy documentation: `numpy.linalg.eigvalsh` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html>, `numpy.linalg.cond` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html>, `numpy.linalg.lstsq` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html>, `numpy.gradient` <https://numpy.org/doc/stable/reference/generated/numpy.gradient.html>, `numpy.meshgrid` <https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html>, and `numpy.isclose` <https://numpy.org/doc/stable/reference/generated/numpy.isclose.html>.
- SciPy documentation: `scipy.integrate.dblquad`, a numerical double integral <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.dblquad.html>, and `scipy.optimize.check_grad` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.check_grad.html> and `scipy.optimize.approx_fprime` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.approx_fprime.html>, library versions of the gradient check you write in this topic.
- Matplotlib documentation: `contour` <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html> and `quiver` <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html>, for the contour and gradient-field plots in the notebook.

## Where this topic is used later

- [Gradient descent and its variants](../gradient-descent-and-its-variants/README.md)
- [Newton and quasi-Newton methods](../newton-and-quasi-newton-methods/README.md)
- [Constrained optimization and KKT](../constrained-optimization-and-kkt/README.md)
- [Probability distributions](../../probability-and-statistics/probability-distributions/README.md)
- [Backpropagation](../../../deep-learning/backpropagation/README.md)
- [Optimization](../../../deep-learning/optimization/README.md)
- [Generative models](../../../deep-learning/generative-models/README.md)
