# Exercises: Vectors and Matrices

Eight problems, easy to harder. Try each on paper first, then check yourself in the [notebook](../notebook.ipynb) or with NumPy. Worked answers are in [solutions.md](solutions.md), so resist peeking until you have had a real go.

Back to the [topic](../README.md).

## Warm-up

**1. Add and scale.** Let $\mathbf{v} = (2, 1)$ and $\mathbf{w} = (-1, 3)$. Compute $\mathbf{v} + \mathbf{w}$, $3\mathbf{v}$, and $2\mathbf{v} - \mathbf{w}$.

**2. Hit a target.** Find numbers $c$ and $d$ so that $c\,(1, 2) + d\,(3, 1) = (7, 4)$.

**3. Length.** (a) Find $\|(3, 4)\|$. (b) Find $\|(1, 2, 2)\|$. (c) Find a vector of length 1 pointing in the same direction as $(3, 4)$.

## Dot products

**4. Perpendicular or not?** (a) Compute $(1, 2, 3) \cdot (4, -5, 2)$. What does the answer say about the two vectors? (b) Find the number $k$ that makes $(k, 2, 1)$ perpendicular to $(2, -1, 4)$.

**5. Angles and similarity.** (a) Find the angle between $(1, 1, 0)$ and $(0, 1, 1)$. (b) A search query is $\mathbf{q} = (3, 1)$. Three documents have vectors $\mathbf{d}_1 = (6, 2)$, $\mathbf{d}_2 = (1, 3)$, $\mathbf{d}_3 = (-3, -1)$. Compute the cosine similarity of each with $\mathbf{q}$ and rank them. What do you notice about $\mathbf{d}_1$, even though it is twice as long as $\mathbf{q}$?

## Matrices

**6. Two views.** Let
$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}, \qquad \mathbf{x} = \begin{bmatrix} 1 \\ -1 \end{bmatrix}.
$$
(a) What is the shape of $A\mathbf{x}$? (b) Compute $A\mathbf{x}$ using the row view (dot products). (c) Compute it again using the column view (a linear combination of the columns) and check that you get the same answer.

**7. Order matters.** Let
$$
A = \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix}, \qquad B = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}.
$$
(a) Compute $AB$ and $BA$. Are they equal? (b) Matrix $B$ is a very simple machine. What does $B\mathbf{x}$ do to a vector $\mathbf{x} = (x_1, x_2)$? (c) A $2 \times 3$ matrix is multiplied by a $3 \times 4$ matrix. What is the shape of the product? Can you multiply a $3 \times 2$ matrix by a $3 \times 4$ matrix?

## Putting it together

**8. A tiny neural network layer.** A layer computes $\mathbf{y} = \text{ReLU}(W\mathbf{x} + \mathbf{b})$, where ReLU replaces every negative number by $0$ and leaves the others unchanged. Let
$$
W = \begin{bmatrix} 1 & 0 & -1 \\ 2 & 1 & 0 \end{bmatrix}, \qquad
\mathbf{b} = \begin{bmatrix} 1 \\ -10 \end{bmatrix}, \qquad
\mathbf{x} = \begin{bmatrix} 3 \\ 2 \\ 1 \end{bmatrix}.
$$
(a) What are the shapes of $\mathbf{x}$ and $\mathbf{y}$? (b) Compute $\mathbf{y}$. (c) Put two inputs side by side as the columns of a matrix $X = \begin{bmatrix} 3 & 0 \\ 2 & 1 \\ 1 & 2 \end{bmatrix}$. Compute $WX$, add $\mathbf{b}$ to each column, and apply ReLU. Which single matrix operation processed both inputs at once?
