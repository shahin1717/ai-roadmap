# Resources: Singular Value Decomposition

Titles, authors, chapter pointers, and free links only. Items marked *(library)* are good books that are not free to download; borrow or buy them if you can.

## Read

| Source | Where to look | Covers |
|---|---|---|
| Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* ([free PDF](https://mml-book.com)) | Ch. 4.5 (Singular Value Decomposition); Ch. 4.6 (Matrix Approximation) | The SVD and its geometry, the rank-$k$ approximation, and why it is the best one, with ML in mind |
| Strang, *Introduction to Linear Algebra*, 5th ed. *(library)* | Ch. 7.1 (Image Processing by Linear Algebra); Ch. 7.2 (Bases and Matrices in the SVD); Ch. 7.3 (Principal Component Analysis (PCA by the SVD)); Ch. 7.4 (The Geometry of the SVD) | Low-rank images, the four subspaces from the SVD, PCA, the rotate-stretch-rotate picture, and the pseudoinverse |
| Strang, *Introduction to Linear Algebra*, 6th ed. *(library)* | Ch. 7.1 (Singular Values and Singular Vectors); Ch. 7.2 (Image Processing by Linear Algebra); Ch. 7.3 (Principal Component Analysis (PCA by the SVD)); Ch. 4.5 (The Pseudoinverse of a Matrix) | The newer edition, with the pseudoinverse in its own section |
| Strang, *Linear Algebra and Learning from Data* *(library)* | Part I, Section I.8 (Singular Values and Singular Vectors in the SVD); Section I.9 (Principal Components and the Best Low Rank Matrix); Part III, Section III.3 (Rapidly Decaying Singular Values) | A short account aimed at machine learning, including the best low-rank matrix |
| Aggarwal, *Linear Algebra and Optimization for Machine Learning* *(library)* | Ch. 7 (Singular Value Decomposition) | The SVD from the linear algebra and the optimization points of view, with applications |
| Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* ([arXiv](https://arxiv.org/abs/2106.09685)) | The whole paper, once you have read the deep-learning topics | Where a low-rank update $BA$ is used in practice |

## Watch

- MIT OpenCourseWare, *18.06 Linear Algebra* (Gilbert Strang): Lecture 29 (Singular value decomposition): <https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-29-singular-value-decomposition/>

## Try

- NumPy documentation: `numpy.linalg.svd` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html>, `numpy.linalg.pinv` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html>, `numpy.linalg.matrix_rank` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html>, `numpy.linalg.cond` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html>, and `numpy.linalg.lstsq` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html>.
- SciPy documentation: `scipy.linalg.svd` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html>, and `scipy.sparse.linalg.svds`, which finds only the top few singular values of a large matrix: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html>
- scikit-learn documentation: `PCA` <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html> (computed with an SVD of the centered data) and `TruncatedSVD` <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html> (SVD without centering, used for sparse matrices such as term-document counts).

## Where this topic is used later

- [Review and project](../review-and-project/README.md)
- [Dimensionality reduction](../../../machine-learning/dimensionality-reduction/README.md)
- [Regularization and hyperparameter tuning](../../../machine-learning/regularization-and-hyperparameter-tuning/README.md)
- [Text representations](../../../deep-learning/text-representations/README.md)
- [GPT and efficient LLMs](../../../deep-learning/gpt-and-efficient-llms/README.md)
