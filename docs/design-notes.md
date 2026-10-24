# Design notes

Decisions about structure and content, and why we made them.

## Decisions

### Getting started comes first

Beginners need a working Python, Jupyter, and NumPy setup before the math notebooks make sense.

### CNN architectures before detection, segmentation, and pose

Detection, segmentation, and pose models are built on top of backbones like ResNet, so learners should meet the backbones first.

### Papers are matched to the topic where they are taught

Each paper note sits in the topic that actually introduces its idea, not in a reading list for some other week.

### Review weeks become review-and-project topics

This is a self-study guide, so there are no exams or grading. Review topics end with a small project instead.

### Graph neural networks get a full topic

Graphs show up in chemistry, social networks, and recommendations, and the spectral view links back to eigenvalues.

### Deep-learning explainability links back to machine learning

Explainability is taught once in machine-learning/explainability-and-error-analysis; the deep-learning side adds only what is new.

### MLOps includes cloud training and cost management

Shipping a model means paying for compute, so cost is part of the engineering story.

### Only README.md and resources.md are required per topic

Notebooks, exercises, assignments, paper notes, and labs are added only where they add real value.

### Concept ownership

The first topic where a concept appears teaches it fully. Later topics give a short recap, link back, and cover only what is new.

### Software engineering is a placeholder

It stays empty until its syllabus is provided.

## Concept ownership map

The owner topic teaches the concept fully. Later topics give a short recap, link back to the owner, and cover only what is new.

| Concept | Owner topic | Recapped in |
|---|---|---|
| Vectors, dot product, matrix multiplication | [vectors-and-matrices](../math/linear-algebra/vectors-and-matrices/README.md) | every later topic |
| Eigenvalues and eigenvectors | [eigenvalues-and-eigenvectors](../math/linear-algebra/eigenvalues-and-eigenvectors/README.md) | machine-learning/dimensionality-reduction, deep-learning/graph-neural-networks |
| SVD | [singular-value-decomposition](../math/linear-algebra/singular-value-decomposition/README.md) | machine-learning/dimensionality-reduction, deep-learning/gpt-and-efficient-llms (LoRA) |
| Least squares | [projections-and-least-squares](../math/linear-algebra/projections-and-least-squares/README.md) | machine-learning/linear-regression |
| Derivatives and the chain rule | [derivatives-and-taylor-series](../math/calculus-and-optimization/derivatives-and-taylor-series/README.md) | math/calculus-and-optimization/multivariable-calculus, deep-learning/backpropagation |
| Gradients and the Hessian | [multivariable-calculus](../math/calculus-and-optimization/multivariable-calculus/README.md) | deep-learning/backpropagation, deep-learning/optimization |
| Gradient descent, SGD, momentum, Adam | [gradient-descent-and-its-variants](../math/calculus-and-optimization/gradient-descent-and-its-variants/README.md) | machine-learning/linear-regression, deep-learning/optimization |
| Lagrange multipliers and KKT | [constrained-optimization-and-kkt](../math/calculus-and-optimization/constrained-optimization-and-kkt/README.md) | machine-learning/support-vector-machines |
| Probability distributions (Bernoulli, Gaussian, ...) | [probability-distributions](../math/probability-and-statistics/probability-distributions/README.md) | machine-learning/logistic-regression-and-naive-bayes, machine-learning/density-estimation-and-gmm |
| Bayes' theorem | [bayesian-probability](../math/probability-and-statistics/bayesian-probability/README.md) | machine-learning/logistic-regression-and-naive-bayes |
| MLE | [estimation-and-mle](../math/probability-and-statistics/estimation-and-mle/README.md) | machine-learning/linear-regression, deep-learning/neural-network-basics (loss functions) |
| Markov chains | [markov-chains-and-sampling](../math/probability-and-statistics/markov-chains-and-sampling/README.md) | machine-learning/reinforcement-learning-intro |
| Train/validation/test split, cross-validation, metrics | [knn-and-model-evaluation](../machine-learning/knn-and-model-evaluation/README.md) | every later ML and DL topic |
| Bias-variance trade-off | [knn-and-model-evaluation](../machine-learning/knn-and-model-evaluation/README.md) | machine-learning/regularization-and-hyperparameter-tuning |
| Regularization (L1, L2, weight decay) | [regularization-and-hyperparameter-tuning](../machine-learning/regularization-and-hyperparameter-tuning/README.md) | deep-learning/regularization |
| Data augmentation | [regularization](../deep-learning/regularization/README.md) | deep-learning/augmentation-and-self-supervised-learning |
| PCA | [dimensionality-reduction](../machine-learning/dimensionality-reduction/README.md) | deep-learning/text-representations |
| Explainability (SHAP, LIME, error analysis) | [explainability-and-error-analysis](../machine-learning/explainability-and-error-analysis/README.md) | deep-learning/alignment-and-responsible-ai |
| Attention | [attention-and-seq2seq](../deep-learning/attention-and-seq2seq/README.md) | deep-learning/transformers, deep-learning/vision-transformers-and-multimodal |
| Transfer learning and fine-tuning | [cnn-architectures-and-transfer-learning](../deep-learning/cnn-architectures-and-transfer-learning/README.md) | deep-learning/pretrained-language-models, deep-learning/gpt-and-efficient-llms |
| Embeddings | [text-representations](../deep-learning/text-representations/README.md) | deep-learning/retrieval-augmented-generation, deep-learning/vision-transformers-and-multimodal |
