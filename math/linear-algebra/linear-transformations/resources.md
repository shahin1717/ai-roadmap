# Resources: Linear Transformations

Titles, authors, chapter pointers, and free links only. Items marked *(library)* are good books that are not free to download; borrow or buy them if you can.

## Read

| Source | Where to look | Covers |
|---|---|---|
| Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* ([free PDF](https://mml-book.com)) | Ch. 2.7 (Linear Mappings); Ch. 2.8 (Affine Spaces); Ch. 3.9 (Rotations) | Linear maps, their matrices, change of basis, affine mappings, and rotations, with ML in mind |
| Strang, *Introduction to Linear Algebra*, 5th ed. *(library)* | Ch. 8.1 (The Idea of a Linear Transformation); Ch. 8.2 (The Matrix of a Linear Transformation); Ch. 8.3 (The Search for a Good Basis) | The classic treatment: linear maps, kernel and range, the matrix of a map, and choosing a basis to make it simple |
| Strang, *Introduction to Linear Algebra*, 6th ed. *(library)* | Ch. 8.1 (The Idea of a Linear Transformation); Ch. 8.2 (The Matrix of a Linear Transformation); Ch. 8.3 (The Search for a Good Basis) | The newer edition of the same chapter |
| Strang, *Linear Algebra and Learning from Data* *(library)* | Part IV, Section IV.5 (Toeplitz Matrices and Shift Invariant Filters); Part VII, Section VII.2 (Convolutional Neural Nets) | Convolution written as a matrix, and where it appears in networks |
| Aggarwal, *Linear Algebra and Optimization for Machine Learning* *(library)* | Ch. 2 (Linear Transformations and Linear Systems), especially 2.2 (The Geometry of Matrix Multiplication) | Rotations, scalings, reflections, and shears as matrices, with machine-learning examples |

## Watch

- MIT OpenCourseWare, *18.06 Linear Algebra* (Gilbert Strang): Lecture 30 (Linear transformations and their matrices): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-30-linear-transformations-and-their-matrices/> and Lecture 31 (Change of basis; image compression): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-31-change-of-basis-image-compression/>
- 3Blue1Brown, *Essence of Linear Algebra*, *Linear transformations and matrices*: <https://www.3blue1brown.com/lessons/linear-transformations>, *Change of basis*: <https://www.3blue1brown.com/lessons/change-of-basis>, and *Abstract vector spaces* (the derivative as a linear map): <https://www.3blue1brown.com/lessons/abstract-vector-spaces>

## Try

- NumPy documentation: `numpy.linalg.inv` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html>, `numpy.linalg.matrix_power` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_power.html>, and `numpy.convolve` <https://numpy.org/doc/stable/reference/generated/numpy.convolve.html>.
- NumPy documentation, polynomials (see `polyder` for the derivative used in the notebook): <https://numpy.org/doc/stable/reference/routines.polynomials.polynomial.html>
- SciPy documentation: `scipy.linalg.null_space` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.null_space.html>, `scipy.linalg.orth` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orth.html>, `scipy.linalg.toeplitz` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.toeplitz.html>, and `scipy.ndimage.affine_transform` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html>, which warps an image with a matrix.
- SymPy documentation, *Matrices*, and *Polynomials* (the notebook uses `sympy.diff` to check the derivative matrix): <https://docs.sympy.org/latest/modules/matrices/matrices.html> and <https://docs.sympy.org/latest/modules/polys/basics.html>
- Matplotlib, *Quick start guide*, for the plots in the notebook: <https://matplotlib.org/stable/users/explain/quick_start.html>

## Where this topic is used later

- [Singular value decomposition](../singular-value-decomposition/README.md)
- [Review and project](../review-and-project/README.md)
- [Multivariable calculus](../../calculus-and-optimization/multivariable-calculus/README.md)
- [Convolutional networks](../../../deep-learning/convolutional-networks/README.md)
- [Transformers](../../../deep-learning/transformers/README.md)
- [Backpropagation](../../../deep-learning/backpropagation/README.md)
- [Dimensionality reduction](../../../machine-learning/dimensionality-reduction/README.md)
