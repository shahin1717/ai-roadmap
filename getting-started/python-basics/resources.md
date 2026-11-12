# Resources: Python Basics

Titles, authors, chapter pointers, and free links only. Everything listed here is free to read online.

## Read

| Source | Where to look | Covers |
|---|---|---|
| *The Python Tutorial*, Python Software Foundation ([docs.python.org](https://docs.python.org/3/tutorial/)) | Ch. 3 (An Informal Introduction to Python); Ch. 4 (More Control Flow Tools: `if`, `for`, `range`, defining functions); Ch. 5 (Data Structures: lists, tuples, dictionaries, comprehensions); Ch. 6 (Modules); Ch. 8 (Errors and Exceptions) | The official, authoritative walk through the language; read it after this topic for the details we skipped |
| Sweigart, *Automate the Boring Stuff with Python*, 2nd ed. ([free online](https://automatetheboringstuff.com/)) | Ch. 1 (Python Basics); Ch. 2 (Flow Control); Ch. 3 (Functions); Ch. 4 (Lists); Ch. 5 (Dictionaries and Structuring Data) | A friendly, example-driven beginner book; the best next step if you want more practice with the core language |
| Downey, *Think Python*, 3rd ed. ([free online](https://allendowney.github.io/ThinkPython/)) | Ch. 1 (Programming as a way of thinking); Ch. 2 (Variables and Statements); the chapters on functions, conditionals and loops, and lists | A slower, more conceptual introduction to programming itself |
| NumPy documentation, *NumPy: the absolute basics for beginners* ([free online](https://numpy.org/doc/stable/user/absolute_beginners.html)) | The whole page; especially "Array fundamentals", "Indexing and slicing", "Basic array operations", "Broadcasting", and "Generating random numbers" | The official beginner guide to arrays, shapes, and the operations used throughout this roadmap |
| VanderPlas, *Python Data Science Handbook* ([free online](https://jakevdp.github.io/PythonDataScienceHandbook/)) | Ch. 2 (Introduction to NumPy): "The Basics of NumPy Arrays", "Computation on NumPy Arrays: Universal Functions", "Aggregations: Min, Max, and Everything In Between", "Computation on Arrays: Broadcasting", "Comparisons, Masks, and Boolean Logic"; Ch. 4 (Visualization with Matplotlib): "Simple Line Plots", "Simple Scatter Plots", "Multiple Subplots" | A clear tour of NumPy and Matplotlib in a notebook style, with many short examples |
| NumPy documentation, *Broadcasting* ([free online](https://numpy.org/doc/stable/user/basics.broadcasting.html)) | The whole page | The precise rules behind adding a row or a number to a table |
| NumPy documentation, *Random Generator* ([free online](https://numpy.org/doc/stable/reference/random/generator.html)) | The introduction and the methods `random`, `normal`, `integers` | The recommended way to draw random numbers, and what a seed does |
| Matplotlib documentation, *Quick start guide* ([free online](https://matplotlib.org/stable/users/explain/quick_start.html)) | "A simple example", "Parts of a Figure", "Coding styles", "Labelling plots", and "Working with multiple Figures and Axes" | Figures, axes, line and scatter plots, labels, and subplots |

## Try

- The [notebook](notebook.ipynb) in this topic, then the [exercises](exercises/problems.md) and the [assignment](assignment.md).
- In any Python session, `help(name)` (or `name?` in a notebook) shows the documentation for a function or module. Get into the habit of asking it.

## Where this topic is used later

- [Vectors and Matrices](../../math/linear-algebra/vectors-and-matrices/README.md), the next stop, and every notebook after it
- [Setup](../setup/README.md), if a library is missing or an import fails
