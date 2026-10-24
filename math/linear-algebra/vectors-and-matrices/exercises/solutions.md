# Solutions: Vectors and Matrices

Worked answers to the [problems](problems.md). Back to the [topic](../README.md).

## 1. Add and scale

- $\mathbf{v} + \mathbf{w} = (2 + (-1),\ 1 + 3) = (1, 4)$
- $3\mathbf{v} = (6, 3)$
- $2\mathbf{v} - \mathbf{w} = (4, 2) - (-1, 3) = (5, -1)$

## 2. Hit a target

Write it as two equations, one per entry:
$$
c + 3d = 7, \qquad 2c + d = 4.
$$
From the second, $d = 4 - 2c$. Put that in the first: $c + 12 - 6c = 7$, so $c = 1$ and $d = 2$.
Check: $1\cdot(1,2) + 2\cdot(3,1) = (1,2) + (6,2) = (7,4)$. ✓

## 3. Length

- (a) $\sqrt{3^2 + 4^2} = \sqrt{25} = 5$.
- (b) $\sqrt{1 + 4 + 4} = \sqrt{9} = 3$.
- (c) Divide by the length: $(3/5,\ 4/5)$. Its length is $\sqrt{9/25 + 16/25} = 1$.

## 4. Perpendicular or not?

- (a) $1\cdot 4 + 2\cdot(-5) + 3\cdot 2 = 4 - 10 + 6 = 0$. A zero dot product means the vectors are perpendicular (orthogonal).
- (b) $(k, 2, 1)\cdot(2, -1, 4) = 2k - 2 + 4 = 2k + 2$. Set it to zero: $k = -1$.

## 5. Angles and similarity

- (a) The dot product is $1\cdot 0 + 1\cdot 1 + 0\cdot 1 = 1$. Each vector has length $\sqrt{2}$, so $\cos\theta = 1/(\sqrt2\cdot\sqrt2) = 1/2$ and $\theta = 60^\circ$.
- (b) $\|\mathbf{q}\| = \sqrt{10}$.
  - $\mathbf{d}_1$: dot $= 18 + 2 = 20$, length $\sqrt{40}$, cosine $= 20/(\sqrt{10}\sqrt{40}) = 20/20 = 1$.
  - $\mathbf{d}_2$: dot $= 3 + 3 = 6$, length $\sqrt{10}$, cosine $= 6/10 = 0.6$.
  - $\mathbf{d}_3$: dot $= -9 - 1 = -10$, length $\sqrt{10}$, cosine $= -10/10 = -1$.

  Ranking: $\mathbf{d}_1$, then $\mathbf{d}_2$, then $\mathbf{d}_3$. Document $\mathbf{d}_1$ is exactly $2\mathbf{q}$, so it points the same way and scores a perfect 1: cosine similarity ignores length and only compares direction. Document $\mathbf{d}_3$ points exactly the opposite way.

## 6. Two views

- (a) $A$ is $3\times 2$ and $\mathbf{x}$ has 2 entries, so the result has 3 entries: a $3\times 1$ vector.
- (b) Row view: $1\cdot 1 + 2\cdot(-1) = -1$, then $3 - 4 = -1$, then $5 - 6 = -1$. So $A\mathbf{x} = (-1, -1, -1)$.
- (c) Column view: $1\cdot(1,3,5) + (-1)\cdot(2,4,6) = (1-2,\ 3-4,\ 5-6) = (-1,-1,-1)$. Same answer. ✓

## 7. Order matters

- (a)
$$
AB = \begin{bmatrix} 1\cdot0+2\cdot1 & 1\cdot1+2\cdot0 \\ 0\cdot0+1\cdot1 & 0\cdot1+1\cdot0 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 1 & 0 \end{bmatrix},
\qquad
BA = \begin{bmatrix} 0\cdot1+1\cdot0 & 0\cdot2+1\cdot1 \\ 1\cdot1+0\cdot0 & 1\cdot2+0\cdot1 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ 1 & 2 \end{bmatrix}.
$$
They are different, so $AB \ne BA$.
- (b) $B\mathbf{x} = (x_2, x_1)$: it swaps the two coordinates (a reflection across the diagonal line).
- (c) $(2\times3)(3\times4)$ gives a $2\times4$ matrix. A $3\times2$ matrix times a $3\times4$ matrix is **not** defined: the inner numbers, 2 and 3, do not match.

## 8. A tiny neural network layer

- (a) $\mathbf{x}$ has 3 entries and $W$ has 2 rows, so $\mathbf{y}$ has 2 entries.
- (b) $W\mathbf{x} = (1\cdot3 + 0\cdot2 + (-1)\cdot1,\ 2\cdot3 + 1\cdot2 + 0\cdot1) = (2, 8)$. Adding $\mathbf{b}$ gives $(3, -2)$. ReLU turns $-2$ into $0$, so $\mathbf{y} = (3, 0)$.
- (c) The second column of $X$ is $(0,1,2)$, and $W(0,1,2) = (-2, 1)$, so
$$
WX = \begin{bmatrix} 2 & -2 \\ 8 & 1 \end{bmatrix}.
$$
Adding $\mathbf{b} = (1, -10)$ to each column gives $\begin{bmatrix} 3 & -1 \\ -2 & -9 \end{bmatrix}$, and ReLU gives
$$
\begin{bmatrix} 3 & 0 \\ 0 & 0 \end{bmatrix}.
$$
The first column matches part (b). One **matrix-matrix multiplication** processed both inputs at once, which is how real networks handle whole batches. (Many libraries store each example as a *row* instead of a column; the idea is the same, only the bookkeeping is flipped.)
