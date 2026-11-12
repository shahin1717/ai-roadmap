# Assignment: Does Walking More Mean Sleeping Better?

Summarise and plot a small, made-up health-tracker dataset using lists, functions, and NumPy. It takes about 45-60 minutes and needs only NumPy and Matplotlib. Back to the [topic](README.md).

## The setup

A fictional person wore a fitness tracker for two weeks, starting on a Monday. Each day the tracker recorded the number of steps, the hours of sleep that night, and a self-rated mood from 1 (low) to 10 (high). These numbers are invented for this exercise. Copy them into a notebook:

```python
days  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
         "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
steps = [4200, 5100, 3900, 6300, 7200, 11800, 9500,
         4800, 5600, 4100, 6800, 7900, 12400, 3100]
sleep = [6.1, 6.5, 5.9, 7.0, 7.2, 8.1, 7.8,
         6.3, 6.8, 6.0, 7.1, 7.4, 8.4, 6.4]
mood  = [5, 6, 4, 6, 7, 8, 8,
         5, 6, 5, 7, 7, 9, 5]
```

The question: on days with more steps, does this person tend to sleep longer?

## Tasks

1. **Build the arrays.** Convert `steps`, `sleep`, and `mood` into NumPy arrays and print the shape of each. Check that all three have 14 entries.
2. **Write a summary function.** Write a function `summarise(values)` with a docstring that returns a dictionary with the keys `"mean"`, `"min"`, `"max"`, and `"std"` (the standard deviation). Call it on each array and print one tidy line per quantity with an f-string, for example `steps: mean 6621, min 3100, max 12400`.
3. **Weekends versus weekdays.** Use a list comprehension on `days` to make a list of `True`/`False` values that says whether each day is a weekend day (`"Sat"` or `"Sun"`), then turn it into a NumPy array to use as a mask. Compute the average steps and the average sleep for weekend days and for weekdays. Print all four numbers.
4. **Measure the relationship.** Compute the correlation between steps and sleep with `np.corrcoef(steps, sleep)[0, 1]`. It is a number between -1 and 1: close to 1 means the two rise together, close to 0 means no pattern, close to -1 means one rises as the other falls. Then fit a straight line with `slope, intercept = np.polyfit(steps, sleep, 1)` (you will learn how this works in the [least squares](../../math/linear-algebra/projections-and-least-squares/README.md) topic) and use it to predict the sleep for a day with 8,000 steps. Write the predicted value with an f-string rounded to one decimal.
5. **Draw it.** Make one figure with two side-by-side subplots (`plt.subplots(1, 2, figsize=(10, 4))`).
   - Left: a line plot of steps against the day number 1 to 14 (`np.arange(1, 15)`), with weekend days marked by a different coloured scatter on top of the line.
   - Right: a scatter plot of steps (horizontal) against sleep (vertical), with the fitted line drawn through it.
   - Both plots need a title and axis labels, and the right-hand plot needs a legend that names the fitted line.
6. **Say what you found.** In 3-5 sentences, answer the question above. Mention the correlation, how weekends differ, and one reason to be careful about the conclusion (hint: 14 days is a small sample, and a high step count on a weekend might come with other differences, such as having more free time).

## What to hand in

A short notebook or script that runs top to bottom, with the figure, plus your written answer to task 6.

## Stretch (optional)

Use the fitted line to simulate more data: make a seeded generator `rng = np.random.default_rng(7)`, draw 200 step counts with `rng.integers(3000, 13000, size=200)`, compute the predicted sleep for each from your line, and add noise with `rng.normal(0, 0.4, size=200)`. Plot the simulated points next to the real ones. Do they look like the real data? Then change the seed and describe what changes in the plot and what stays the same.
