# Resources: Gradient Descent and its Variants

Titles, authors, chapter pointers, and free links only. Items marked *(library)* are good books that are not free to download; borrow or buy them if you can.

## Read

| Source | Where to look | Covers |
|---|---|---|
| Deisenroth, Faisal & Ong, *Mathematics for Machine Learning* ([free PDF](https://mml-book.com)) | Ch. 7 (Continuous Optimization): 7.1 Optimization Using Gradient Descent, with 7.1.1 Step-size, 7.1.2 Gradient Descent With Momentum, and 7.1.3 Stochastic Gradient Descent | Sections 1, 2, 5, 6, and 7 of this page, in a short and mathematical style |
| Aggarwal, *Linear Algebra and Optimization for Machine Learning* *(library)* | Ch. 4 (Optimization Basics: A Machine Learning View): 4.2.1 Univariate Optimization (with Why We Need Gradient Descent, Convergence of Gradient Descent, and The Divergence Problem); 4.2.3 Multivariate Optimization; 4.4.1 Checking Gradient Correctness with Finite Differences; 4.4.2 Learning Rate Decay and Bold Driver; 4.5.2 Stochastic Gradient Descent; 4.5.5 The Importance of Feature Preprocessing | Sections 1 to 6, the gradient check, and feature scaling |
| Same book *(library)* | Ch. 5 (Advanced Optimization Solutions): 5.2 Challenges in Gradient-Based Optimization (Local Optima and Flat Regions, Differential Curvature, Revisiting Feature Normalization); 5.3 Adjusting First-Order Derivatives for Descent (Momentum-Based Learning, AdaGrad, RMSProp, Adam) | Sections 3, 4, and 7 to 9, and why curvature makes valleys hard |
| Same book *(library)* | Ch. 4: 4.7 Linear Regression: Optimization with Numerical Targets; 4.8.3 Logistic Regression | The two models used as worked examples in the AI/ML section |
| Strang, *Linear Algebra and Learning from Data* *(library)* | Part VI (Optimization): VI.4 Gradient Descent Toward the Minimum; VI.5 Stochastic Gradient Descent and ADAM | Sections 1 to 3 and 5 to 9, written for people who know linear algebra |
| Hass, Heil & Weir (based on the work of George B. Thomas, Jr.), *Thomas' Calculus: Early Transcendentals*, 13th ed. *(library)* | Ch. 14: 14.7 Extreme Values and Saddle Points | A second look at critical points and saddles from section 4 |
| Ruder, *An overview of gradient descent optimization algorithms* ([blog post](https://www.ruder.io/optimizing-gradient-descent/), [arXiv:1609.04747](https://arxiv.org/abs/1609.04747)) | The whole article | A tour of batch, stochastic, and mini-batch descent, momentum, Nesterov, AdaGrad, RMSProp, and Adam |
| Kingma & Ba, *Adam: A Method for Stochastic Optimization* ([arXiv:1412.6980](https://arxiv.org/abs/1412.6980)) | The algorithm box and the section on bias correction | The original statement of Adam, with the defaults used in section 9 |
| Loshchilov & Hutter, *Decoupled Weight Decay Regularization* ([arXiv:1711.05101](https://arxiv.org/abs/1711.05101)) | The introduction | Where AdamW comes from, the forward pointer in section 9 |

## Watch and explore

- Goh, *Why Momentum Really Works*, Distill, 2017 <https://distill.pub/2017/momentum/>. Interactive pictures of the heavy-ball idea and of how momentum changes the stable range of step sizes (section 7).
- 3Blue1Brown, *Gradient descent, how neural networks learn* <https://www.3blue1brown.com/lessons/gradient-descent>. A visual first look at descending a loss with many parameters.
- CS231n course notes, *Neural Networks Part 3: Learning and Evaluation* <https://cs231n.github.io/neural-networks-3/>. The sections on gradient checks, parameter updates (momentum, Nesterov, per-parameter adaptive methods), and annealing the learning rate.

## Try

- PyTorch documentation, `torch.optim` <https://pytorch.org/docs/stable/optim.html>, and the learning-rate scheduler `CosineAnnealingLR` <https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html>. These are the library versions of the optimisers and schedules you write by hand here.
- NumPy documentation: `numpy.linalg.lstsq` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html>, `numpy.linalg.eigvalsh` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html>, `numpy.linalg.cond` <https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html>, `numpy.random.Generator.permutation` <https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.permutation.html>, `numpy.isclose` <https://numpy.org/doc/stable/reference/generated/numpy.isclose.html>, and `numpy.allclose` <https://numpy.org/doc/stable/reference/generated/numpy.allclose.html>.
- SciPy documentation: `scipy.optimize.minimize` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>, the library solver to compare with your own loops, and `scipy.optimize.check_grad` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.check_grad.html>, a library version of the gradient check.
- Matplotlib documentation: `contour` <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html>, for the path plots in the notebook.

## Where this topic is used later

- [Newton and quasi-Newton methods](../newton-and-quasi-newton-methods/README.md)
- [Conjugate gradient and trust region](../conjugate-gradient-and-trust-region/README.md)
- [Constrained optimization and KKT](../constrained-optimization-and-kkt/README.md)
- [Linear regression](../../../machine-learning/linear-regression/README.md)
- [Logistic regression and naive Bayes](../../../machine-learning/logistic-regression-and-naive-bayes/README.md)
- [Optimization (deep learning)](../../../deep-learning/optimization/README.md)
- [Transformers](../../../deep-learning/transformers/README.md)
