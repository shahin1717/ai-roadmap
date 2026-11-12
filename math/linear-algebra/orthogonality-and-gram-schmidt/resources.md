# Resources: Orthogonality and Gram-Schmidt

Titles, authors, chapter pointers, and free links only. Items marked *(library)* are good books that are not free to download; borrow or buy them if you can.

## Read

| Source | Where to look | Covers |
|---|---|---|
| Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* ([free PDF](https://mml-book.com)) | Ch. 3.4 (Angles and Orthogonality); Ch. 3.5 (Orthonormal Basis); Ch. 3.6 (Orthogonal Complement) | Orthogonality, orthonormal bases, and complements, with ML in mind |
| Strang, *Introduction to Linear Algebra*, 5th ed. *(library)* | Ch. 4.1 (Orthogonality of the Four Subspaces); Ch. 4.4 (Orthonormal Bases and Gram-Schmidt) | The perpendicular pairs of subspaces, orthonormal bases, and the A = QR factorization |
| Strang, *Introduction to Linear Algebra*, 6th ed. *(library)* | Ch. 4.1 (Orthogonality of Vectors and Subspaces); Ch. 4.4 (Orthonormal Bases and Gram-Schmidt) | The newer edition of the same material |
| Strang, *Linear Algebra and Learning from Data* *(library)* | Part I, Section I.5 (Orthogonal Matrices and Subspaces) | A short account of orthogonal matrices and subspaces, with applications to learning |
| Aggarwal, *Linear Algebra and Optimization for Machine Learning* *(library)* | Ch. 2.7 (Generating Orthogonal Basis Sets), especially 2.7.1 (Gram-Schmidt Orthogonalization and QR Decomposition) and 2.7.2 (QR Decomposition) | Gram-Schmidt and QR with machine-learning examples |

## Watch

- MIT OpenCourseWare, *18.06 Linear Algebra* (Gilbert Strang): Lecture 14 (Orthogonal Vectors and Subspaces): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-14-orthogonal-vectors-and-subspaces/> and Lecture 17 (Orthogonal Matrices and Gram-Schmidt): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-17-orthogonal-matrices-and-gram-schmidt/>
- 3Blue1Brown, *Essence of Linear Algebra*: *Dot products and duality* <https://www.3blue1brown.com/lessons/dot-products>, and *Linear transformations and matrices* <https://www.3blue1brown.com/lessons/linear-transformations> (rotations are the standard example).

## Try

- NumPy documentation, `numpy.linalg.qr`: <https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html>
- SciPy documentation, `scipy.linalg.qr` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.qr.html>, and `scipy.linalg.hilbert`, the nearly dependent test matrix used in the notebook: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.hilbert.html>
- SciPy documentation, `scipy.stats.ortho_group`, which draws random orthogonal matrices: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ortho_group.html>
- PyTorch documentation, `torch.nn.init`, the weight initialization functions (it includes an orthogonal initializer): <https://pytorch.org/docs/stable/nn.init.html>
- Matplotlib, *Quick start guide*, for the plots in the notebook: <https://matplotlib.org/stable/users/explain/quick_start.html>

## Where this topic is used later

- [Projections and least squares](../projections-and-least-squares/README.md)
- [Eigenvalues and eigenvectors](../eigenvalues-and-eigenvectors/README.md)
- [Linear transformations](../linear-transformations/README.md)
- [Singular value decomposition](../singular-value-decomposition/README.md)
- [Neural network basics](../../../deep-learning/neural-network-basics/README.md)
- [Dimensionality reduction](../../../machine-learning/dimensionality-reduction/README.md)
