# Resources: Matrix Operations and LU Decomposition

Titles, authors, chapter pointers, and free links only. Items marked *(library)* are good books that are not free to download; borrow or buy them if you can.

## Read

| Source | Where to look | Covers |
|---|---|---|
| Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* ([free PDF](https://mml-book.com)) | Ch. 2.2 (Matrices), which includes inverses and transposes; Ch. 4.1 (Determinant and Trace) | Matrix algebra, the inverse, the transpose, and the determinant, written with ML in mind |
| Strang, *Introduction to Linear Algebra*, 5th ed. *(library)* | Ch. 2.4 (Rules for Matrix Operations); Ch. 2.5 (Inverse Matrices); Ch. 2.6 (Elimination = Factorization: A = LU); Ch. 2.7 (Transposes and Permutations); Ch. 5.1 (The Properties of Determinants); Ch. 5.2 (Permutations and Cofactors); Ch. 5.3 (Cramer's Rule, Inverses, and Volumes) | The classic treatment of every idea on this page |
| Strang, *Introduction to Linear Algebra*, 6th ed. *(library)* | Ch. 1.4 (Matrix Multiplication AB and CR); Ch. 2.2 (Elimination Matrices and Inverse Matrices); Ch. 2.3 (Matrix Computations and A = LU); Ch. 2.4 (Permutations and Transposes); Ch. 5.1 (3 by 3 Determinants and Cofactors); Ch. 5.2 (Computing and Using Determinants); Ch. 5.3 (Areas and Volumes by Determinants) | The newer edition, with the four views of a product up front |
| Strang, *Linear Algebra and Learning from Data* *(library)* | Part I, Section I.2 (Matrix-Matrix Multiplication AB); Section I.4 (Elimination and A = LU) | A short, ML-oriented account of products and LU |
| Aggarwal, *Linear Algebra and Optimization for Machine Learning* *(library)* | Ch. 1.2.4 (Matrix Powers, Polynomials, and the Inverse); Ch. 2.5.1 (LU Decomposition); Ch. 2.5.3 (Application: Matrix Inversion); Ch. 3.2 (Determinants) | Inverses, LU, and determinants with machine-learning examples |

## Watch

- MIT OpenCourseWare, *18.06 Linear Algebra* (Gilbert Strang): Lecture 3 (Multiplication and Inverse Matrices): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-3-multiplication-and-inverse-matrices/>; Lecture 4 (Factorization into A = LU): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-4-factorization-into-a-lu/>; Lecture 5 (Transposes, Permutations, Spaces R^n): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-5-transposes-permutations-spaces-r-n/>; Lecture 18 (Properties of Determinants): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-18-properties-of-determinants/>; Lecture 19 (Determinant Formulas and Cofactors): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-19-determinant-formulas-and-cofactors/>
- 3Blue1Brown, *Essence of Linear Algebra*: the chapters on matrix multiplication as composition, on the determinant, and on inverse matrices, column space and null space: <https://www.3blue1brown.com/topics/linear-algebra>

## Try

- NumPy documentation: `numpy.linalg.inv` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html>, `numpy.linalg.det` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html>, and `numpy.linalg.slogdet` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html> (the log-determinant used for Gaussian densities).
- SciPy documentation: `scipy.linalg.lu` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu.html> (returns $P$, $L$, $U$ with $A = PLU$), `scipy.linalg.lu_factor` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_factor.html>, and `scipy.linalg.lu_solve` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_solve.html> for factor once, solve many.

## Where this topic is used later

- [Vector spaces and subspaces](../vector-spaces-and-subspaces/README.md)
- [Eigenvalues and eigenvectors](../eigenvalues-and-eigenvectors/README.md)
- [Projections and least squares](../projections-and-least-squares/README.md)
- [Probability distributions](../../probability-and-statistics/probability-distributions/README.md)
- [Neural network basics](../../../deep-learning/neural-network-basics/README.md)
