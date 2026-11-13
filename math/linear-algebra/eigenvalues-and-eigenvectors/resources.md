# Resources: Eigenvalues and Eigenvectors

Titles, authors, chapter pointers, and free links only. Items marked *(library)* are good books that are not free to download; borrow or buy them if you can.

## Read

| Source | Where to look | Covers |
|---|---|---|
| Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* ([free PDF](https://mml-book.com)) | Ch. 4.2 (Eigenvalues and Eigenvectors); Ch. 4.3 (Cholesky Decomposition); Ch. 4.4 (Eigendecomposition and Diagonalization) | Eigenvalues, diagonalization, and the symmetric positive definite case, with ML in mind |
| Strang, *Introduction to Linear Algebra*, 5th ed. *(library)* | Ch. 6.1 (Introduction to Eigenvalues); Ch. 6.2 (Diagonalizing a Matrix); Ch. 6.4 (Symmetric Matrices); Ch. 6.5 (Positive Definite Matrices); Ch. 11.3 (Iterative Methods and Preconditioners), which includes the power method and the inverse power method | The classic treatment of everything on this page |
| Strang, *Introduction to Linear Algebra*, 6th ed. *(library)* | Ch. 6.1 (Introduction to Eigenvalues: Ax = λx); Ch. 6.2 (Diagonalizing a Matrix); Ch. 6.3 (Symmetric Positive Definite Matrices) | The newer edition, with symmetric and positive definite matrices in one section |
| Strang, *Linear Algebra and Learning from Data* *(library)* | Part I, Section I.6 (Eigenvalues and Eigenvectors); Section I.7 (Symmetric Positive Definite Matrices) | A short account aimed at machine learning |
| Aggarwal, *Linear Algebra and Optimization for Machine Learning* *(library)* | Ch. 3.3 (Diagonalizable Transformations and Eigenvectors), including 3.3.7 (Symmetric Matrices) and 3.3.8 (Positive Semidefinite Matrices); Ch. 3.5 (Numerical Algorithms for Finding Eigenvectors), including 3.5.2 (The Power Method for Finding Dominant Eigenvectors) | Eigen-decompositions, symmetric matrices, and the power method, with machine-learning examples |

## Watch

- MIT OpenCourseWare, *18.06 Linear Algebra* (Gilbert Strang): Lecture 21 (Eigenvalues and eigenvectors): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-21-eigenvalues-and-eigenvectors/>; Lecture 22 (Diagonalization and powers of A): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-22-diagonalization-and-powers-of-a/>; Lecture 25 (Symmetric matrices and positive definiteness): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-25-symmetric-matrices-and-positive-definiteness/>; Lecture 27 (Positive definite matrices and minima): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-27-positive-definite-matrices-and-minima/>
- 3Blue1Brown, *Essence of Linear Algebra*, *Eigenvectors and eigenvalues*: <https://www.3blue1brown.com/lessons/eigenvalues>

## Try

- NumPy documentation: `numpy.linalg.eig` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html>, `numpy.linalg.eigh` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html>, and `numpy.linalg.cholesky` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html>.
- SciPy documentation: `scipy.linalg.eigh` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html> and `scipy.sparse.linalg.eigsh`, which finds a few eigenvalues of a large symmetric matrix: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>
- SymPy documentation, *Matrices* (see `eigenvals`, `eigenvects`, and `is_diagonalizable`), for exact eigenvalues: <https://docs.sympy.org/latest/modules/matrices/matrices.html>
- Matplotlib, *Quick start guide*, for the plots in the notebook: <https://matplotlib.org/stable/users/explain/quick_start.html>

## Where this topic is used later

- [Linear transformations](../linear-transformations/README.md)
- [Singular value decomposition](../singular-value-decomposition/README.md)
- [Markov chains and sampling](../../probability-and-statistics/markov-chains-and-sampling/README.md)
- [Dimensionality reduction](../../../machine-learning/dimensionality-reduction/README.md)
- [Clustering](../../../machine-learning/clustering/README.md)
- [Recurrent networks](../../../deep-learning/recurrent-networks/README.md)
- [Graph neural networks](../../../deep-learning/graph-neural-networks/README.md)
