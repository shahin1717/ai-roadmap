# Resources: Linear Equations and Gaussian Elimination

Titles, authors, chapter pointers, and free links only. Items marked *(library)* are good books that are not free to download; borrow or buy them if you can.

## Read

| Source | Where to look | Covers |
|---|---|---|
| Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* ([free PDF](https://mml-book.com)) | Ch. 2.1 (Systems of Linear Equations); Ch. 2.3 (Solving Systems of Linear Equations) | Systems written as matrices, the three outcomes, and elimination, with machine-learning examples |
| Strang, *Introduction to Linear Algebra*, 5th ed. *(library)* | Ch. 2.1 (Vectors and Linear Equations); Ch. 2.2 (The Idea of Elimination); Ch. 2.3 (Elimination Using Matrices) | The row and column pictures, pivots, back substitution, and elimination matrices, picture-first |
| Strang, *Introduction to Linear Algebra*, 6th ed. *(library)* | Ch. 2.1 (Elimination and Back Substitution); first part of Ch. 2.2 (Elimination Matrices and Inverse Matrices) | The same ideas in the newer edition; the inverse part belongs to the next topic |
| Strang, *Linear Algebra and Learning from Data* *(library)* | Part I, Section I.4 (Elimination and A = LU), the elimination part; Part II, Section II.1 (Numerical Linear Algebra) | A short account of elimination, and why pivoting and sensitivity matter in practice |
| Aggarwal, *Linear Algebra and Optimization for Machine Learning* *(library)* | Ch. 2.5 (The Row Echelon Form of a Matrix), especially 2.5.4 (Application: Solving a System of Linear Equations); Ch. 2.9 (Ill-Conditioned Matrices and Systems) | Elimination and solving systems from a machine-learning angle, and how sensitivity to errors arises |

## Watch

- MIT OpenCourseWare, *18.06 Linear Algebra* (Gilbert Strang), Lecture 1 (The Geometry of Linear Equations): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-1-the-geometry-of-linear-equations/> and Lecture 2 (Elimination with Matrices): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-2-elimination-with-matrices/>
- 3Blue1Brown, *Essence of Linear Algebra*, chapter 7 (Inverse matrices, column space and null space), for a visual view of what solving $A\mathbf{x} = \mathbf{b}$ means: <https://www.3blue1brown.com/topics/linear-algebra>

## Try

- NumPy documentation, `numpy.linalg.solve`, the library routine for $A\mathbf{x} = \mathbf{b}$: <https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html>
- NumPy documentation, `numpy.vander`, which builds the Vandermonde matrix used in the polynomial examples: <https://numpy.org/doc/stable/reference/generated/numpy.vander.html>
- Matplotlib, *Quick start guide*, for the plots in the notebook: <https://matplotlib.org/stable/users/explain/quick_start.html>

## Where this topic is used later

- [Matrix operations and LU decomposition](../matrix-operations-and-lu-decomposition/README.md)
- [Vector spaces and subspaces](../vector-spaces-and-subspaces/README.md)
- [Projections and least squares](../projections-and-least-squares/README.md)
- [Linear regression](../../../machine-learning/linear-regression/README.md)
