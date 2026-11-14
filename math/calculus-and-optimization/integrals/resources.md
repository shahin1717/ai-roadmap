# Resources: Integrals

Titles, authors, chapter pointers, and free links only. Items marked *(library)* are good books that are not free to download; borrow or buy them if you can.

## Read

| Source | Where to look | Covers |
|---|---|---|
| Hass, Heil & Weir (based on the work of George B. Thomas, Jr.), *Thomas' Calculus: Early Transcendentals*, 13th ed. *(library)* | Ch. 5 (Integrals): 5.1 Area and Estimating with Finite Sums; 5.2 Sigma Notation and Limits of Finite Sums; 5.3 The Definite Integral (including the average value of a continuous function); 5.4 The Fundamental Theorem of Calculus; 5.5 Indefinite Integrals and the Substitution Method; 5.6 Definite Integral Substitutions and the Area Between Curves | Sections 1 to 4 and 7 of this page: Riemann sums, the definite integral and its properties, the fundamental theorem, substitution, area between curves, average value |
| Same book *(library)* | Ch. 4.8 (Antiderivatives); Ch. 6 (Applications of Definite Integrals), 6.1 Volumes Using Cross-Sections | The antiderivative table and the $+C$; volumes by slicing and discs |
| Same book *(library)* | Ch. 8 (Techniques of Integration): 8.1 Using Basic Integration Formulas; 8.2 Integration by Parts; 8.3 Trigonometric Integrals; 8.5 Integration of Rational Functions by Partial Fractions; 8.7 Numerical Integration; 8.8 Improper Integrals; 8.9 Probability | By parts, the two techniques this page only names (trigonometric integrals and partial fractions), the trapezoid and Simpson rules with their errors, improper integrals, and probability densities as integrals |
| Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* ([free PDF](https://mml-book.com)) | Ch. 6.2 (Discrete and Continuous Probabilities); 6.4 (Summary Statistics and Independence); 6.5 (Gaussian Distribution) | A pointer for the AI side: probability density functions, cumulative distribution functions, and expected values for continuous variables, which are the integrals of this page, plus the Gaussian |
| Wasserman, *All of Statistics* *(library)* | Ch. 2 (Random Variables): 2.2 Distribution Functions and Probability Functions, 2.4 Some Important Continuous Random Variables; Ch. 3 (Expectation): 3.1 Expectation of a Random Variable | A compact second view of densities, the CDF, and expectation as an integral |

## Watch

- MIT OpenCourseWare, *18.01SC Single Variable Calculus*, Unit 3, *The Definite Integral and Its Applications*: Part A, *Definition of the Definite Integral and First Fundamental Theorem* <https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/unit-3-the-definite-integral-and-its-applications/part-a-definition-of-the-definite-integral-and-first-fundamental-theorem/>; Part B, *Second Fundamental Theorem, Areas, Volumes* <https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/unit-3-the-definite-integral-and-its-applications/part-b-second-fundamental-theorem-areas-volumes/>; Part C, *Average Value, Probability and Numerical Integration* <https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/unit-3-the-definite-integral-and-its-applications/part-c-average-value-probability-and-numerical-integration/>.
- The same course, Unit 4, *Part B: Partial Fractions, Integration by Parts, Arc Length, and Surface Area* <https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/unit-4-techniques-of-integration/part-b-partial-fractions-integration-by-parts-arc-length-and-surface-area/>, and Unit 5, *Part A: L'Hospital's Rule and Improper Integrals* <https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/pages/unit-5-exploring-the-infinite/part-a-lhospitals-rule-and-improper-integrals/>. The full lecture series is at <https://ocw.mit.edu/courses/18-01-single-variable-calculus-fall-2006/>.
- 3Blue1Brown, *Essence of Calculus*: *Integration and the fundamental theorem of calculus* <https://www.3blue1brown.com/lessons/integration>. A picture-first explanation of why the area function and the derivative undo each other.

## Practice and more explanations

- Paul's Online Math Notes, *Calculus I*, the Integrals chapter: <https://tutorial.math.lamar.edu/Classes/CalcI/IntegralsIntro.aspx>. Its sections include Indefinite Integrals, Computing Indefinite Integrals, Substitution Rule, Area Problem, Definition of the Definite Integral, Computing Definite Integrals, and Substitution Rule for Definite Integrals, for example <https://tutorial.math.lamar.edu/Classes/CalcI/DefnOfDefiniteIntegral.aspx> and <https://tutorial.math.lamar.edu/Classes/CalcI/ComputingDefiniteIntegrals.aspx>. The applications chapter has <https://tutorial.math.lamar.edu/Classes/CalcI/AreaBetweenCurves.aspx>, <https://tutorial.math.lamar.edu/Classes/CalcI/AvgFcnValue.aspx>, and <https://tutorial.math.lamar.edu/Classes/CalcI/VolumeWithRings.aspx>.
- The same notes in *Calculus II*: Integration by Parts <https://tutorial.math.lamar.edu/Classes/CalcII/IntegrationByParts.aspx>, Approximating Definite Integrals <https://tutorial.math.lamar.edu/Classes/CalcII/ApproximatingDefIntegrals.aspx>, and Improper Integrals <https://tutorial.math.lamar.edu/Classes/CalcII/ImproperIntegrals.aspx>.
- Khan Academy, the integral calculus course: <https://www.khanacademy.org/math/integral-calculus>. Short videos and practice questions.

## Try

- SciPy documentation: `scipy.integrate.quad` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html>, `scipy.integrate.trapezoid` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.trapezoid.html>, `scipy.integrate.simpson` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html>, `scipy.integrate.cumulative_trapezoid` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cumulative_trapezoid.html>, and the user guide on integration <https://docs.scipy.org/doc/scipy/tutorial/integrate.html>.
- SymPy documentation: the *Calculus* part of the introductory tutorial <https://docs.sympy.org/latest/tutorials/intro-tutorial/calculus.html> and the integrals module reference <https://docs.sympy.org/latest/modules/integrals/integrals.html>.
- NumPy documentation: `numpy.isclose` <https://numpy.org/doc/stable/reference/generated/numpy.isclose.html>, for comparing floats instead of `==`.
- Matplotlib, *Quick start guide*, for the plots in the notebook: <https://matplotlib.org/stable/users/explain/quick_start.html>

## Where this topic is used later

- [Multivariable calculus](../multivariable-calculus/README.md)
- [Random variables](../../probability-and-statistics/random-variables/README.md)
- [Probability distributions](../../probability-and-statistics/probability-distributions/README.md)
- [Markov chains and sampling](../../probability-and-statistics/markov-chains-and-sampling/README.md)
- [KNN and model evaluation](../../../machine-learning/knn-and-model-evaluation/README.md)
