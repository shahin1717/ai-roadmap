# Solutions: Setup

Worked answers to the [problems](problems.md). The outputs below were produced by running the commands on Linux with Python 3.11; folder names and version numbers on your machine will differ, and `/home/you` stands for your own home folder. Windows and macOS differences are noted where they matter. Back to the [topic](../README.md).

## 1. Find your way around

```bash
pwd
mkdir setup-practice
cd setup-practice
echo "I can use the terminal" > hello.txt
ls
cat hello.txt
```

```text
/home/you
hello.txt
I can use the terminal
```

The first line is the folder you started in (on Windows, something like `C:\Users\you`). After `cd setup-practice` you are inside the new folder, `echo ... > hello.txt` writes the sentence into a new file, `ls` shows that file, and `cat` prints it. If `cd` says the folder does not exist, you mistyped its name or ran `mkdir` somewhere else; use `pwd` and `ls` to check.

## 2. Which Python?

| | macOS and Linux | Windows PowerShell |
|---|---|---|
| (a) version | `python3 --version` | `python --version` (or `py --version`) |
| (b) location | `which python3` | `Get-Command python` (or `where.exe python`) |

```text
Python 3.11.16
/usr/bin/python3
```

(c) If the version is not 3.11, you have two choices from the README: install Python 3.11 from python.org and use `venv`, or use Miniconda, which installs the version you ask for inside an environment. If it is 3.10 or 3.12, most things will still work, but 3.11 is the tested version.

## 3. Make and enter an environment

With `venv` (macOS and Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip --version
```

```text
pip 24.0 from /home/you/setup-practice/.venv/lib/python3.11/site-packages/pip (python 3.11)
```

On Windows use `py -3.11 -m venv .venv` and `.venv\Scripts\Activate.ps1`. With conda:

```bash
conda create --name practice python=3.11
conda activate practice
python -m pip --version
```

Two things changed. The prompt now starts with `(.venv)` (or `(practice)` with conda, replacing `(base)`), and the path printed by `pip --version` points *inside* the environment (`.../.venv/lib/...` or `.../envs/practice/...`) instead of a system folder. Both are the evidence that activation worked.

## 4. Install and inspect a package

```bash
python -m pip install numpy
python -m pip show numpy
python -m pip list
```

```text
Successfully installed numpy-2.4.6
Name: numpy
Version: 2.4.6
...
Location: /home/you/setup-practice/.venv/lib/python3.11/site-packages
...
Package    Version
---------- -------
numpy      2.4.6
pip        24.0
setuptools 79.0.1
```

(`...` marks lines left out for space. A newer NumPy may be available by the time you read this.) `pip show` prints a package's version and its `Location`. The location must be inside your environment folder; if it is not, you installed into the wrong Python.

## 5. Pin and unpin

```bash
python -m pip install numpy==1.26.4
python -m pip show numpy
python -m pip install --upgrade numpy
python -m pip show numpy
```

```text
Successfully installed numpy-1.26.4
Version: 1.26.4
Successfully installed numpy-2.4.6
Version: 2.4.6
```

`==` asks for exactly one version. pip removes the other one for you. Without `--upgrade`, `pip install numpy` would leave an already-installed version alone.

## 6. Requirements file and snapshot

(a) and (b):

```bash
echo numpy > requirements.txt        # one line per package; a text editor works too
echo matplotlib >> requirements.txt  # >> adds to the file instead of replacing it
python -m pip install -r requirements.txt
```

(c):

```bash
python -m pip freeze > requirements-lock.txt
cat requirements-lock.txt
```

```text
contourpy==1.3.3
cycler==0.12.1
fonttools==4.68.0
kiwisolver==1.5.1
matplotlib==3.11.2
numpy==2.4.6
packaging==26.3
pillow==13.0.0
pyparsing==3.3.3
python-dateutil==4.0.0
tzdata==2026.6
```

(d) Your file named only the two packages you wanted. Matplotlib itself needs `pillow`, `kiwisolver`, `pyparsing`, and more; those are its **dependencies**, and pip installed them automatically. `pip freeze` lists the whole installed set with exact versions, so the lock file is longer and reproduces the environment exactly.

## 7. Break it and rebuild it

```bash
deactivate
rm -r .venv                  # Windows PowerShell: Remove-Item -Recurse -Force .venv
python3 -m venv .venv        # Windows: py -3.11 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -c "import numpy, matplotlib; print(numpy.__version__, matplotlib.__version__)"
python -m pip freeze | diff - requirements-lock.txt && echo identical
```

```text
2.4.6 3.11.2
identical
```

The `diff` command compares two inputs line by line and prints nothing when they match, so `echo identical` runs only if the rebuilt environment equals the snapshot. (`diff` is a macOS/Linux command; in PowerShell, compare by eye, or run `python -m pip freeze` and look at the two lists.) With conda, delete with `conda remove --name practice --all`, create a new one, and reinstall with `python -m pip install -r requirements-lock.txt`. Being able to do this on demand is the whole point of an environment.

## 8. What went wrong?

- **(a)** The notebook's **kernel** uses a different Python from the one you installed into. In a cell, run `import sys; print(sys.executable)` to see which Python the kernel is. Switch the kernel to your environment (VS Code: **Select Kernel**; Jupyter: the Kernel menu, after registering it with `python -m ipykernel install --user --name practice`). Or install from inside the notebook with `%pip install numpy`, which always targets the running kernel.
- **(b)** PowerShell's execution policy blocks the activation script. Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, confirm, and activate again. Alternatively, activate in Command Prompt with `.venv\Scripts\activate.bat`.
- **(c)** The system Python belongs to the operating system, which protects it. Do not force it (no `sudo pip`, no `--break-system-packages`). Create an environment and install there.
- **(d)** The file is not in the current folder (or the name is mistyped). Run `pwd` and `ls` to see where you are, then `cd` to the repo folder, or pass the path, for example `python -m pip install -r path/to/requirements-math.txt`.

## 9. No install, no problem

(a) The notebook prints a Python version (the exact one is Colab's choice and changes over time) and `Running in Colab : True`, and the environment name is Colab's own, not yours.

(b) In a new cell:

```python
%pip install pyjokes
import pyjokes
print(pyjokes.get_joke())
```

pip prints its usual install messages, and then a joke appears.

(c) After the runtime is deleted and you reconnect, `import pyjokes` fails with `ModuleNotFoundError`: Colab gave you a fresh temporary computer, so the package you installed is gone. Put the `%pip install ...` lines in the first cells of the notebook, so they run again each session. Anything else you want to keep (files, trained models) must be saved to Drive or downloaded before the session ends.
