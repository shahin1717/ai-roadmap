# Exercises: Python Basics

Eleven problems, easy to harder. Type each solution into a notebook cell and run it; do not just read. Worked answers are in [solutions.md](solutions.md), so resist peeking until you have had a real go. If you get stuck, reread the matching section of the [topic](../README.md) or try the [notebook](../notebook.ipynb).

Back to the [topic](../README.md).

## Warm-up

**1. Temperature report.** A sensor reads 21.5 degrees Celsius. Store it in a variable, convert it to Fahrenheit with $F = 9C/5 + 32$, and print one line that reads `21.5 C is 70.7 F` using an f-string with one decimal place for each number.

**2. Slicing a list.** Let `data = [3, 8, 1, 9, 4, 7, 2]`. Using indexing and slicing, produce: (a) the first three items; (b) the last two items; (c) every second item, starting with the first; (d) the list reversed; (e) the position of the largest value (hint: lists have an `.index(value)` method).

**3. Counting words.** Given the sentence `"the cat and the dog and the bird"`, use a dictionary and a loop to count how often each word appears. Print the dictionary. (Hint: `sentence.split()` turns a sentence into a list of words, and `d.get(word, 0)` returns 0 for a word you have not seen yet.)

## Control flow and functions

**4. How often is the model right?** A classifier outputs the probabilities `[0.9, 0.3, 0.6, 0.2, 0.8]` for five examples, and the true labels are `[1, 0, 0, 0, 1]`. Turn each probability into a predicted label (1 if the probability is at least 0.5, otherwise 0), then use a loop and `zip` to count how many predictions match the true labels. Print the fraction that are correct.

**5. Rescale to the range 0 to 1.** Write a function `min_max(values)` that takes a list of numbers and returns a new list where the smallest number becomes 0, the largest becomes 1, and the others are scaled in between: $(v - \text{min}) / (\text{max} - \text{min})$. Give it a docstring. Test it on `[2, 4, 6, 10]`. Then extend it so that a list whose numbers are all equal returns a list of zeros instead of failing.

**6. Sigmoid with a comprehension.** The **sigmoid** function squeezes any number into the range between 0 and 1: $\sigma(x) = 1 / (1 + e^{-x})$. Write a function `sigmoid(x)` using `math.exp`, then use a list comprehension to apply it to `[-2, 0, 2]`. Round the results to three decimals when you print them. Which input gives exactly 0.5?

## NumPy

**7. Indexing a grid.** Build `A = np.arange(20).reshape(4, 5)`. Extract: (a) the second column; (b) the last row; (c) the top-left 2-by-2 block; (d) all entries greater than 10, as a flat array; (e) a copy of `A` in which every entry greater than 15 has been replaced by 0, with the original `A` unchanged.

**8. Loop versus vectorized.** The mean squared error between predictions and targets is the average of $(\text{prediction} - \text{target})^2$. For `predictions = np.array([2.5, 0.0, 2.1, 7.8])` and `targets = np.array([3.0, -0.5, 2.0, 7.5])`: (a) compute it with a `for` loop over the entries; (b) compute it again with a single array expression; (c) check that the two answers agree (use `np.isclose`, since floats are approximate).

**9. Standardise the columns.** Create a data matrix with
```python
rng = np.random.default_rng(1)
X = rng.normal(loc=[10, 50, 200], scale=[1, 5, 20], size=(100, 3))
```
so that each of the 3 columns has a very different average and spread. Subtract each column's mean and divide by its standard deviation, in one array expression. Then verify that the new column means are close to 0 and the new column standard deviations are close to 1. What is the shape of `X.mean(axis=0)`, and why?

**10. A batch through a layer.** A simple neural-network layer turns a batch of inputs `X` (shape `(4, 3)`: 4 examples, 3 features) into outputs with $Z = XW + \mathbf{b}$, followed by ReLU, which replaces negative numbers by 0 (use `np.maximum(0, Z)`). Make `X`, `W` of shape `(3, 2)`, and `b` of shape `(2,)` with a seeded generator (`rng.normal`). (a) Compute the output and print its shape. (b) Explain why adding `b`, which has only 2 numbers, to a `(4, 2)` matrix works. (c) Without running it, say whether `W @ X` would work, and why.

## Debugging

**11. Find the bug.** Each snippet below fails or gives a wrong result. For each one, say what the problem is, then fix it.

```python
# (a)
scores = [10, 20, 30]
print(scores[3])
```

```python
# (b)
def double_all(values):
    doubled = [v * 2 for v in values]
    print(doubled)

result = double_all([1, 2, 3])
print(sum(result))
```

```python
# (c)
import numpy as np

weights = np.array([0.2, 0.5, 0.3])
inputs = np.ones((2, 3))
print(weights @ inputs)
```
