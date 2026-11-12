# Exercises: Independence, Basis, and Dimension

Ten problems, easy to harder. Do the first nine on paper, then check yourself in the [notebook](../notebook.ipynb) or with NumPy. The last one is a coding task. Worked answers are in [solutions.md](solutions.md). Give each problem a real attempt before you open them.

Back to the [topic](../README.md).

## Independence and span

**1. Independent or dependent?** Decide for each list. For a dependent list, write a nontrivial combination that equals the zero vector. (a) $(1,2)$, $(3,4)$. (b) $(1,2)$, $(2,4)$. (c) $(1,0,0)$, $(0,1,0)$, $(1,1,0)$. (d) $(1,0)$, $(0,1)$, $(5,7)$.

**2. The matrix test.** Put the vectors $(1,2,3)$, $(4,5,6)$, $(7,8,9)$ in the columns of a matrix. Find its rank and its nullspace. Are the columns independent? What does the nullspace say about the columns?

**3. Spans.** (a) Describe the span of $(1,2,1)$ and $(2,4,2)$. (b) Describe the span of $(1,0,1)$ and $(0,1,1)$ by an equation, and decide whether $(3,2,5)$ is in it. (c) What is the span of $(1,0,0)$, $(0,1,0)$, $(0,0,1)$, and $(7,7,7)$, and how many of the four vectors are really needed?

## Basis and dimension

**4. Coordinates.** The vectors $(1,2)$ and $(2,3)$ form a basis of $\mathbb{R}^2$. Find the coordinates of $\mathbf{v} = (4,7)$ in this basis by solving a small system. Then explain why $(1,2)$ and $(2,4)$ cannot be a basis of $\mathbb{R}^2$.

**5. A basis for a plane.** Find a basis for the plane $x - y + 2z = 0$ in $\mathbb{R}^3$ (hint: it is the nullspace of a $1 \times 3$ matrix) and state its dimension. Check that each basis vector satisfies the equation.

**6. A basis from vectors.** Let $\mathbf{v}_1 = (1,1,0)$, $\mathbf{v}_2 = (2,2,0)$, $\mathbf{v}_3 = (3,4,1)$, $\mathbf{v}_4 = (1,0,-1)$. Row reduce the matrix with these columns, give a basis for their span and its dimension, and write each dependent vector as a combination of the basis vectors.

## The four subspaces

**7. All four at once.** Let
```math
A = \begin{bmatrix} 1 & 2 & 0 & 3 \\ 0 & 0 & 1 & 4 \\ 1 & 2 & 1 & 7 \end{bmatrix}.
```
Find $R$, the rank, and a basis for each of the four fundamental subspaces: the column space, the row space, the nullspace, and the left nullspace. Check that the dimensions add up as $r + (n - r) = n$ and $r + (m - r) = m$, and check one dot product that shows the row space is perpendicular to the nullspace.

**8. Counting dimensions.** For each shape and rank, give the four dimensions and say whether $A\mathbf{x} = \mathbf{b}$ is solvable for every $\mathbf{b}$ and whether solutions are unique. (a) $5 \times 8$, rank $3$. (b) $6 \times 6$, rank $6$. (c) $4 \times 7$, rank $4$. (d) Why is a $5 \times 8$ matrix of rank $6$ impossible?

## Rank of products, and code

**9. Rank of a product.** (a) Let $A$ be $5 \times 3$ with rank $2$ and $B$ be $3 \times 4$ with rank $3$. What is the largest possible rank of $AB$, and why? (b) A linear network maps $\mathbb{R}^6$ to a hidden layer with $2$ values and then back to $\mathbb{R}^6$, so its matrix is $W_2W_1$ with $W_1$ of size $2 \times 6$ and $W_2$ of size $6 \times 2$. What is the largest possible rank of $W_2W_1$, and what does that say about the outputs?

**10. Write the tools.** Write `is_independent(vectors)`, which returns `True` or `False` and, when the vectors are dependent, a nontrivial combination of them that is zero. Write `basis_for_span(vectors)`, which returns a basis as a matrix whose columns are the pivot columns (use SymPy's `rref` to find the pivot positions). Test them on problems 1, 2, and 6. Finally, check rank-nullity, $r + \dim N(A) = n$, on 100 random matrices of various shapes and ranks made with a seeded generator.
