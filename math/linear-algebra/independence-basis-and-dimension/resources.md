# Resources: Independence, Basis, and Dimension

Titles, authors, chapter pointers, and free links only. Items marked *(library)* are good books that are not free to download; borrow or buy them if you can.

## Read

| Source | Where to look | Covers |
|---|---|---|
| Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* ([free PDF](https://mml-book.com)) | Ch. 2.5 (Linear Independence); Ch. 2.6 (Basis and Rank) | Independence, bases, dimension, and rank, written with ML in mind |
| Strang, *Introduction to Linear Algebra*, 5th ed. *(library)* | Ch. 3.4 (Independence, Basis and Dimension); Ch. 3.5 (Dimensions of the Four Subspaces) | The classic treatment, including the four subspaces and rank-nullity |
| Strang, *Introduction to Linear Algebra*, 6th ed. *(library)* | Ch. 3.4 (Independence, Basis, and Dimension); Ch. 3.5 (Dimensions of the Four Subspaces) | The newer edition of the same material |
| Strang, *Linear Algebra and Learning from Data* *(library)* | Part I, Section I.3 (The Four Fundamental Subspaces) | A short account of the four subspaces and their dimensions |
| Aggarwal, *Linear Algebra and Optimization for Machine Learning* *(library)* | Ch. 2.3 (Vector Spaces and Their Geometry), especially 2.3.1 (Coordinates in a Basis System) and 2.3.3 (Span of a Set of Vectors); Ch. 2.5.2 (Application: Finding a Basis Set); Ch. 2.6 (The Notion of Matrix Rank), including 2.6.1 (Effect of Matrix Operations on Rank) | Spans, bases, rank, and how products affect rank, with machine-learning examples |

## Watch

- MIT OpenCourseWare, *18.06 Linear Algebra* (Gilbert Strang): Lecture 9 (Independence, Basis, and Dimension): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-9-independence-basis-and-dimension/>; Lecture 10 (The Four Fundamental Subspaces): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-10-the-four-fundamental-subspaces/>; Lecture 11 (Matrix Spaces; Rank 1; Small World Graphs): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-11-matrix-spaces-rank-1-small-world-graphs/>
- 3Blue1Brown, *Essence of Linear Algebra*, chapter 2 (Linear combinations, span, and basis vectors): <https://www.3blue1brown.com/lessons/span>

## Try

- NumPy documentation: `numpy.linalg.matrix_rank` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html> and `numpy.linalg.cond` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html>.
- SciPy documentation: `scipy.linalg.null_space` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.null_space.html> and `scipy.linalg.orth` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orth.html>, which return bases for the nullspace and the column space.
- SymPy documentation, *Matrices* (see `rref`), for exact row reduction and pivot columns: <https://docs.sympy.org/latest/modules/matrices/matrices.html>
- Matplotlib, *Quick start guide*, for the plots in the notebook: <https://matplotlib.org/stable/users/explain/quick_start.html>

## Where this topic is used later

- [Orthogonality and Gram-Schmidt](../orthogonality-and-gram-schmidt/README.md)
- [Linear transformations](../linear-transformations/README.md)
- [Eigenvalues and eigenvectors](../eigenvalues-and-eigenvectors/README.md)
- [Singular value decomposition](../singular-value-decomposition/README.md)
- [Linear regression](../../../machine-learning/linear-regression/README.md)
- [Dimensionality reduction](../../../machine-learning/dimensionality-reduction/README.md)
