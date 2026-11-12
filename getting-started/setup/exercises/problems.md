# Exercises: Setup

Nine hands-on tasks, easy to harder. Unlike most exercises, these are done in a terminal and a notebook, not on paper. Do them on your own computer (or, for task 9, in Colab), type the commands yourself, and compare what you see with [solutions.md](solutions.md). Use either `venv` or Miniconda for the environment tasks; the solutions show both.

Back to the [topic](../README.md).

## Warm-up: the terminal and Python

**1. Find your way around.** Open a terminal. (a) Print the folder you are in. (b) Make a folder named `setup-practice` and move into it. (c) Create a text file `hello.txt` that contains the sentence `I can use the terminal` (hint: on macOS, Linux, and PowerShell, `echo "..." > hello.txt` does it). (d) List the folder, then print the file's contents.

**2. Which Python?** (a) Print your Python version. (b) Print the location of the Python program that answered. (c) Is it Python 3.11? If not, what does the README suggest you do next?

## Environments and pip

**3. Make and enter an environment.** Inside `setup-practice`, create an environment (a `.venv` folder with `venv`, or one named `practice` with conda, using Python 3.11 if you can). Activate it. Then prove that you are inside it: show two things that changed, one in the prompt and one in the output of `python -m pip --version`.

**4. Install and inspect a package.** With the environment active, install NumPy using `python -m pip`. Then use pip to show which version you got and where it was installed. Finally list all packages in the environment.

**5. Pin and unpin.** Install NumPy version `1.26.4` exactly, and confirm the version. Then upgrade NumPy to the newest version and confirm it changed.

**6. Requirements file and snapshot.** (a) Create a file `requirements.txt` containing two lines, `numpy` and `matplotlib`. (b) Install from it with one command. (c) Save a snapshot of every installed package, with exact versions, to `requirements-lock.txt`. (d) Look at the lock file: why is it longer than your two-line file?

**7. Break it and rebuild it.** Deactivate the environment and delete it. Create a new one and rebuild it from `requirements-lock.txt` only. Run a one-line Python command that imports NumPy and Matplotlib and prints their versions. Finally, check that the new environment matches the snapshot exactly.

## Diagnosis

**8. What went wrong?** For each symptom, name the most likely cause and the fix.

- (a) You ran `python -m pip install numpy` successfully, but a notebook cell `import numpy` gives `ModuleNotFoundError: No module named 'numpy'`.
- (b) On Windows, `.venv\Scripts\Activate.ps1` ends with `...cannot be loaded because running scripts is disabled on this system`.
- (c) On Linux, `pip install numpy` (no environment active) prints `error: externally-managed-environment`.
- (d) `python -m pip install -r requirements-math.txt` prints `Could not open requirements file: [Errno 2] No such file or directory`.

## Colab

**9. No install, no problem.** Open this topic's [notebook](../notebook.ipynb) in Colab with its badge. (a) Run the checks and note the Python version and whether `Running in Colab` says `True`. (b) In a new cell, install a package that is not preinstalled (for example `%pip install pyjokes`), import it, and call `pyjokes.get_joke()`. (c) Use **Runtime > Disconnect and delete runtime**, reconnect, and run the `import pyjokes` cell again. What happens, and what does that tell you about where to put install commands in a Colab notebook?
