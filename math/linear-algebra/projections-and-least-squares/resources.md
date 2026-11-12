# Resources: Projections and Least Squares

Titles, authors, chapter pointers, and free links only. Items marked *(library)* are good books that are not free to download; borrow or buy them if you can.

## Read

| Source | Where to look | Covers |
|---|---|---|
| Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* ([free PDF](https://mml-book.com)) | Ch. 3.8 (Orthogonal Projections); Ch. 9.2 (Parameter Estimation); Ch. 9.4 (Maximum Likelihood as Orthogonal Projection) | Projections, linear regression, and the link between least squares and projection, with ML in mind |
| Strang, *Introduction to Linear Algebra*, 5th ed. *(library)* | Ch. 4.2 (Projections); Ch. 4.3 (Least Squares Approximations) | The classic, picture-first treatment of projections, the normal equations, and line fitting |
| Strang, *Introduction to Linear Algebra*, 6th ed. *(library)* | Ch. 4.2 (Projections onto Lines and Subspaces); Ch. 4.3 (Least Squares Approximations); Ch. 4.5 (The Pseudoinverse of a Matrix) | The newer edition, with the pseudoinverse for dependent columns |
| Strang, *Linear Algebra and Learning from Data* *(library)* | Part II, Section II.2 (Least Squares: Four Ways) | Four ways to solve least squares and when each is the right one |
| Aggarwal, *Linear Algebra and Optimization for Machine Learning* *(library)* | Ch. 2.8 (An Optimization-Centric View of Linear Systems), including 2.8.1 (Moore-Penrose Pseudoinverse) and 2.8.2 (The Projection Matrix); Ch. 4.7 (Linear Regression: Optimization with Numerical Targets) | Least squares, the projection matrix, and the pseudoinverse, with machine-learning examples |

## Watch

- MIT OpenCourseWare, *18.06 Linear Algebra* (Gilbert Strang): Lecture 15 (Projections onto subspaces): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-15-projections-onto-subspaces/> and Lecture 16 (Projection matrices and least squares): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-16-projection-matrices-and-least-squares/>
- 3Blue1Brown, *Essence of Linear Algebra*, *Dot products and duality*, for the projection view of the dot product: <https://www.3blue1brown.com/lessons/dot-products>

## Try

- NumPy documentation: `numpy.linalg.lstsq` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html> and `numpy.linalg.pinv` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html>.
- SciPy documentation: `scipy.linalg.lstsq` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html> and `scipy.linalg.solve_triangular`, for the back substitution step of a QR solve: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_triangular.html>
- NumPy documentation, `numpy.vander`, which builds the polynomial design matrix: <https://numpy.org/doc/stable/reference/generated/numpy.vander.html>
- Matplotlib, *Quick start guide*, for the plots in the notebook: <https://matplotlib.org/stable/users/explain/quick_start.html>

## Where this topic is used later

- [Eigenvalues and eigenvectors](../eigenvalues-and-eigenvectors/README.md)
- [Singular value decomposition](../singular-value-decomposition/README.md)
- [Estimation and MLE](../../probability-and-statistics/estimation-and-mle/README.md)
- [Linear regression](../../../machine-learning/linear-regression/README.md)
- [Regularization and hyperparameter tuning](../../../machine-learning/regularization-and-hyperparameter-tuning/README.md)
- [Dimensionality reduction](../../../machine-learning/dimensionality-reduction/README.md)
