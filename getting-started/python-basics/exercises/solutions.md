# Solutions: Python Basics

Worked answers to the [problems](problems.md). Every snippet was run, and the output shown is the real output. Your code may look different and still be right: compare the results, not the wording. Back to the [topic](../README.md).

## 1. Temperature report

```python
celsius = 21.5
fahrenheit = celsius * 9 / 5 + 32
print(f"{celsius:.1f} C is {fahrenheit:.1f} F")
```

```text
21.5 C is 70.7 F
```

The `.1f` after the colon asks for one digit after the decimal point. The arithmetic is $21.5 \cdot 9 / 5 + 32 = 38.7 + 32 = 70.7$.

## 2. Slicing a list

```python
data = [3, 8, 1, 9, 4, 7, 2]
print(data[:3])                        # (a) first three
print(data[-2:])                       # (b) last two
print(data[::2])                       # (c) every second item
print(data[::-1])                      # (d) reversed
print(data.index(max(data)))           # (e) position of the largest value
```

```text
[3, 8, 1]
[7, 2]
[3, 1, 4, 2]
[2, 7, 4, 9, 1, 8, 3]
3
```

For (b), `-2:` starts two places from the end and runs to the end. For (e), `max(data)` is 9 and `data.index(9)` says where 9 sits; counting from 0, that is position 3.

## 3. Counting words

```python
sentence = "the cat and the dog and the bird"
counts = {}
for word in sentence.split():
    counts[word] = counts.get(word, 0) + 1
print(counts)
```

```text
{'the': 3, 'cat': 1, 'and': 2, 'dog': 1, 'bird': 1}
```

For each word, `counts.get(word, 0)` returns the current count, or 0 if the word is new, and we store that number plus one. After the loop, `"the"` has been seen 3 times, `"and"` twice, and the others once.

## 4. How often is the model right?

```python
probabilities = [0.9, 0.3, 0.6, 0.2, 0.8]
labels = [1, 0, 0, 0, 1]

predicted = [1 if p >= 0.5 else 0 for p in probabilities]
print(predicted)

correct = 0
for guess, truth in zip(predicted, labels):
    if guess == truth:
        correct += 1
print(correct, "of", len(labels), "correct, fraction", correct / len(labels))
```

```text
[1, 0, 1, 0, 1]
4 of 5 correct, fraction 0.8
```

The third example (probability 0.6, true label 0) is the only miss, so 4 of 5 are right. The expression `1 if p >= 0.5 else 0` is a compact one-line `if`/`else`; you could also write it with a normal `if` block inside a loop.

## 5. Rescale to the range 0 to 1

```python
def min_max(values):
    """Rescale a list of numbers so the smallest becomes 0 and the largest 1.

    If all numbers are equal there is nothing to stretch, so return zeros.
    """
    low = min(values)
    high = max(values)
    if high == low:
        return [0.0 for _ in values]
    return [(v - low) / (high - low) for v in values]

print(min_max([2, 4, 6, 10]))
print(min_max([5, 5, 5]))
```

```text
[0.0, 0.25, 0.5, 1.0]
[0.0, 0.0, 0.0]
```

Here low is 2 and high is 10, so the span is 8: the values become $0/8$, $2/8$, $4/8$, $8/8$. Without the `if`, the list `[5, 5, 5]` would divide by zero and Python would raise a `ZeroDivisionError`. (The underscore `_` is a conventional name for a loop variable you do not use.)

## 6. Sigmoid with a comprehension

```python
import math

def sigmoid(x):
    """Squeeze any number into the range between 0 and 1."""
    return 1 / (1 + math.exp(-x))

outputs = [round(sigmoid(x), 3) for x in [-2, 0, 2]]
print(outputs)
```

```text
[0.119, 0.5, 0.881]
```

The input 0 gives exactly 0.5, because $e^{0} = 1$ and so $1/(1+1) = 0.5$. Large positive inputs approach 1, large negative ones approach 0, and the two printed values are mirror images ($0.119 + 0.881 = 1$). Sigmoid turns a raw score into something that can be read as a probability.

## 7. Indexing a grid

```python
import numpy as np

A = np.arange(20).reshape(4, 5)
print(A)
print(A[:, 1])             # (a) second column
print(A[-1])               # (b) last row
print(A[:2, :2])           # (c) top-left 2x2 block
print(A[A > 10])           # (d) entries greater than 10
B = A.copy()               # (e) copy first, then edit the copy
B[B > 15] = 0
print(B)
print(A[3])                # the original is unchanged
```

```text
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]]
[ 1  6 11 16]
[15 16 17 18 19]
[[0 1]
 [5 6]]
[11 12 13 14 15 16 17 18 19]
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15  0  0  0  0]]
[15 16 17 18 19]
```

Assigning through a mask, `B[B > 15] = 0`, overwrites only the positions where the condition is true. Without `.copy()`, `B` would just be a second name for the same array and `A` would change too.

## 8. Loop versus vectorized

```python
predictions = np.array([2.5, 0.0, 2.1, 7.8])
targets = np.array([3.0, -0.5, 2.0, 7.5])

total = 0.0                                   # (a) loop
for p, t in zip(predictions, targets):
    total += (p - t) ** 2
mse_loop = total / len(predictions)

mse_vec = ((predictions - targets) ** 2).mean()   # (b) one expression

print(mse_loop, mse_vec)
print(np.isclose(mse_loop, mse_vec))              # (c)
```

```text
0.14999999999999997 0.14999999999999997
True
```

The differences are $-0.5, 0.5, 0.1, 0.3$, their squares add up to $0.25 + 0.25 + 0.01 + 0.09 = 0.6$, and dividing by 4 gives $0.15$. Here the two numbers print identically, but on another machine or with other data the last digits can differ, because the two ways add things up in a different order and floats are approximate. `np.isclose` allows for that.

## 9. Standardise the columns

```python
rng = np.random.default_rng(1)
X = rng.normal(loc=[10, 50, 200], scale=[1, 5, 20], size=(100, 3))

Z = (X - X.mean(axis=0)) / X.std(axis=0)

print(np.allclose(Z.mean(axis=0), 0))   # every column mean is (almost) 0
print(np.allclose(Z.std(axis=0), 1))    # every column spread is (almost) 1
print(X.mean(axis=0).shape)
```

```text
True
True
(3,)
```

`X.mean(axis=0)` collapses the 100 rows and leaves one average per column, so its shape is `(3,)`. Broadcasting then subtracts that row of 3 numbers from every row of `X`, and the same trick divides by the 3 standard deviations. We test with `np.allclose` (like `np.isclose`, but it checks every entry and answers with a single `True` or `False`) instead of `==` because floats leave tiny leftovers, such as a mean of `1e-16` instead of exactly 0. Putting every column on the same scale like this matters because many ML methods behave badly when one feature is measured in thousands and another in tenths.

## 10. A batch through a layer

```python
rng = np.random.default_rng(0)
X = rng.normal(size=(4, 3))      # 4 examples, 3 features
W = rng.normal(size=(3, 2))      # maps 3 features to 2 outputs
b = rng.normal(size=2)           # one bias per output

Z = X @ W + b
out = np.maximum(0, Z)           # ReLU
print(out.shape)
print(out.round(3))

try:
    W @ X
except ValueError as error:
    print("W @ X fails with a ValueError; shapes", W.shape, X.shape)
```

```text
(4, 2)
[[0.    0.909]
 [0.638 1.297]
 [0.    0.286]
 [4.108 1.763]]
W @ X fails with a ValueError; shapes (3, 2) (4, 3)
```

(a) `(4, 3) @ (3, 2)` has matching inner sizes (3 and 3), so the result is `(4, 2)`: one row of 2 outputs per example. (b) Broadcasting treats `b` of shape `(2,)` as a row that is added to every row of the `(4, 2)` matrix, so each example gets the same bias. (c) `W @ X` would be `(3, 2) @ (4, 3)`. The inner sizes are 2 and 4, which differ, so NumPy raises a `ValueError`, as the last run shows. This is why one writes `X @ W` for a batch of examples stored as rows. The entries of `out` that are exactly 0 are the ones ReLU clipped.

## 11. Find the bug

**(a)** A list with 3 items has valid positions 0, 1, 2, so `scores[3]` is out of range and raises an `IndexError`. The last item is `scores[2]` or `scores[-1]`.

```python
scores = [10, 20, 30]
try:
    print(scores[3])
except IndexError as error:
    print("IndexError:", error)
print(scores[-1])
```

```text
IndexError: list index out of range
30
```

**(b)** `double_all` prints the doubled list but never returns it, so `result` is `None`, and `sum(None)` raises a `TypeError`. Replace `print(doubled)` by `return doubled` (and print outside the function if you want to see it).

```python
def double_all(values):
    doubled = [v * 2 for v in values]
    return doubled

result = double_all([1, 2, 3])
print(result, sum(result))
```

```text
[2, 4, 6] 12
```

**(c)** `weights` has shape `(3,)` and `inputs` has shape `(2, 3)`. In `weights @ inputs` the vector has 3 entries but must be matched against the first dimension of `inputs`, which is 2, so the shapes do not fit. If the idea is "one score per row of `inputs`", put the matrix first: `inputs @ weights`, shape `(2, 3) @ (3,)` giving `(2,)`.

```python
weights = np.array([0.2, 0.5, 0.3])
inputs = np.ones((2, 3))
try:
    weights @ inputs
except ValueError as error:
    print("ValueError, as expected")
print(inputs @ weights)
```

```text
ValueError, as expected
[1. 1.]
```

Printing `weights.shape` and `inputs.shape` before the product is the quickest way to spot this kind of mistake.
