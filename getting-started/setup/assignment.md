# Assignment: A Reproducible Environment

Build a clean environment for this roadmap, prove it works with a small script, and show that you can rebuild it from a file. It takes about 30-45 minutes and the only thing you hand in is a few lines of text and a small script. Back to the [topic](README.md).

## The setup

Reproducibility means that someone else (or you, next year) can recreate your working conditions and get the same results. You will make that true for this roadmap. Choose Option A (`venv`) or Option B (Miniconda) from the [topic](README.md); you only need one. If you can only use Colab, do tasks 4 and 5 there and write down which steps do not apply.

## Tasks

1. **Create and activate.** Get the repo (see [section 10](README.md#10-get-the-repo-with-git)) and create an environment for it with Python 3.11 if you can. Name it `ai-roadmap`, or use the folder `.venv` inside the repo. Activate it and confirm that `python --version` and `python -m pip --version` point inside it.
2. **Install the math requirements.** From the repo folder, install `requirements-math.txt` with `python -m pip install -r requirements-math.txt`, then run `python -m pip check`.
3. **Run the check script.** Save the script below as `check_env.py`, run it, and fix anything it reports.
4. **Snapshot it.** Save a snapshot with `python -m pip freeze > requirements-lock.txt` (do not commit it to the repo). Note its number of lines with `wc -l requirements-lock.txt` (macOS/Linux) or `(Get-Content requirements-lock.txt).Count` (PowerShell).
5. **Rebuild from the snapshot.** Delete the environment, create a new one, install only from your snapshot, and run `check_env.py` again. The result should be the same.
6. **Open a notebook in the right kernel.** Open this topic's [notebook](notebook.ipynb) in VS Code or Jupyter, select your environment as the kernel, and run it. Compare its "Environment name" with what you expect.
7. **Write it down.** In 4-6 sentences, answer: Which option did you choose and why? What was the one thing that went wrong or surprised you, and how did you find the cause? Why did you rebuild from the snapshot instead of copying the environment folder?

## The check script

Save this as `check_env.py`. It checks the Python version, looks for the packages the math course needs, and exits with a clear message instead of a crash.

```python
import sys
from importlib import metadata
from importlib.util import find_spec

REQUIRED = ["numpy", "scipy", "sympy", "matplotlib", "jupyter"]


def main():
    ok = True
    major, minor = sys.version_info[:2]
    print(f"Python {major}.{minor} at {sys.prefix}")
    if (major, minor) != (3, 11):
        print("  note: this repo is tested with Python 3.11, you have", f"{major}.{minor}")

    for name in REQUIRED:
        if find_spec(name) is None:
            print(f"  MISSING  {name}")
            ok = False
        else:
            print(f"  ok       {name} {metadata.version(name)}")

    print("All good." if ok else "Something is missing: install requirements-math.txt.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

A good result looks like this (versions will differ; the second line appears only if your Python is not 3.11):

```text
Python 3.11 at /home/you/ai-roadmap/.venv
  ok       numpy 2.4.6
  ok       scipy 1.17.1
  ok       sympy 1.14.0
  ok       matplotlib 3.11.2
  ok       jupyter 1.1.1
All good.
```

## What to hand in

The text answer to task 7, the output of `check_env.py` before and after the rebuild, and the number of lines in your snapshot.

## Stretch (optional)

Add a second environment for the machine-learning course: create `ai-roadmap-ml`, install `requirements-ml.txt`, and compare the two snapshots with `diff` (or a text editor) to see exactly which packages the ML layer adds on top of the math layer.
