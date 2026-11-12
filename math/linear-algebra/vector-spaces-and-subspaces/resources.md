# Resources: Vector Spaces and Subspaces

Titles, authors, chapter pointers, and free links only. Items marked *(library)* are good books that are not free to download; borrow or buy them if you can.

## Read

| Source | Where to look | Covers |
|---|---|---|
| Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* ([free PDF](https://mml-book.com)) | Ch. 2.4 (Vector Spaces); Ch. 2.3 (Solving Systems of Linear Equations) | Vector spaces and subspaces, and the particular-plus-general solution of a system, with ML in mind |
| Strang, *Introduction to Linear Algebra*, 5th ed. *(library)* | Ch. 3.1 (Spaces of Vectors); Ch. 3.2 (The Nullspace of A: Solving Ax = 0 and Rx = 0); Ch. 3.3 (The Complete Solution to Ax = b) | The classic, picture-first treatment of everything on this page |
| Strang, *Introduction to Linear Algebra*, 6th ed. *(library)* | Ch. 3.1 (Vector Spaces and Subspaces); Ch. 3.2 (Computing the Nullspace by Elimination: A = CR); Ch. 3.3 (The Complete Solution to Ax = b) | The newer edition, with the same ideas organised around column and row factors |
| Strang, *Linear Algebra and Learning from Data* *(library)* | Part I, Section I.3 (The Four Fundamental Subspaces) | A short account of the column space and nullspace, which continues in the next topic |
| Aggarwal, *Linear Algebra and Optimization for Machine Learning* *(library)* | Ch. 2.3 (Vector Spaces and Their Geometry); Ch. 2.5 (The Row Echelon Form of a Matrix); Ch. 2.6 (The Notion of Matrix Rank) | Subspaces, row reduction, and rank with machine-learning examples |

## Watch

- MIT OpenCourseWare, *18.06 Linear Algebra* (Gilbert Strang): Lecture 5, the vector spaces part (Transposes, Permutations, Spaces R^n): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-5-transposes-permutations-spaces-r-n/>; Lecture 6 (Column Space and Nullspace): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-6-column-space-and-nullspace/>; Lecture 7 (Solving Ax = 0: Pivot Variables, Special Solutions): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-7-solving-ax-0-pivot-variables-special-solutions/>; Lecture 8 (Solving Ax = b: Row Reduced Form R): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-8-solving-ax-b-row-reduced-form-r/>
- 3Blue1Brown, *Essence of Linear Algebra*, chapter 7 (Inverse matrices, column space and null space): <https://www.3blue1brown.com/topics/linear-algebra>

## Try

- NumPy documentation, `numpy.linalg.matrix_rank`: <https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html>
- SciPy documentation, `scipy.linalg.null_space`, an orthonormal basis for the nullspace: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.null_space.html>
- SymPy documentation, *Matrices* (see `rref` and `nullspace`), for exact row reduction with fractions: <https://docs.sympy.org/latest/modules/matrices/matrices.html>
- Matplotlib, *Quick start guide*, for the plots in the notebook: <https://matplotlib.org/stable/users/explain/quick_start.html>

## Where this topic is used later

- [Independence, basis, and dimension](../independence-basis-and-dimension/README.md)
- [Orthogonality and Gram-Schmidt](../orthogonality-and-gram-schmidt/README.md)
- [Projections and least squares](../projections-and-least-squares/README.md)
- [Singular value decomposition](../singular-value-decomposition/README.md)
- [Linear regression](../../../machine-learning/linear-regression/README.md)
