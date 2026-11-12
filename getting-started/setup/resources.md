# Resources: Setup

Titles, authors, chapter pointers, and free links only. Everything listed here is free to read or download.

## Python and pip

| Source | Where to look | Covers |
|---|---|---|
| Python Software Foundation, *Python downloads* ([python.org](https://www.python.org/downloads/)) | The latest release and the full list of versions | Installers for Windows and macOS |
| Python Software Foundation, *Using Python on Windows* ([docs.python.org](https://docs.python.org/3/using/windows.html)) | The sections on the installer and the `py` launcher | What the Windows installer options mean, and how to pick a version with `py` |
| Python Software Foundation, *The Python Tutorial*, Ch. 12 (Virtual Environments and Packages) ([docs.python.org](https://docs.python.org/3/tutorial/venv.html)) | The whole chapter | Creating, activating, and using `venv`, and managing packages with pip |
| Python Software Foundation, *venv: creation of virtual environments* ([docs.python.org](https://docs.python.org/3/library/venv.html)) | "Creating virtual environments" and the activation table | The exact activation command for every shell, including PowerShell |
| Python Packaging Authority, *pip user guide* ([pip.pypa.io](https://pip.pypa.io/en/stable/user_guide/)) | "Requirements Files", "Installing Packages", and the section on proxies | How pip installs, pins versions, and works with requirements files |
| Python Packaging Authority, *Requirements file format* ([pip.pypa.io](https://pip.pypa.io/en/stable/reference/requirements-file-format/)) | The whole page | Version specifiers and the `-r` option inside a requirements file |
| Python Packaging Authority, *pip freeze* ([pip.pypa.io](https://pip.pypa.io/en/stable/cli/pip_freeze/)) | The whole page | Snapshotting an environment |
| Python Packaging User Guide, *Install packages using pip and venv* ([packaging.python.org](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)) | The whole guide | A step-by-step walk through the venv and pip workflow on every system |
| Python Package Index ([pypi.org](https://pypi.org/)) | Search for any package | The catalogue pip downloads from |

## Conda and Miniconda

| Source | Where to look | Covers |
|---|---|---|
| Anaconda, *Miniconda* ([anaconda.com](https://www.anaconda.com/docs/getting-started/miniconda/main)) | The overview page, and the install pages for Windows, macOS, and Linux under "Installing Miniconda" ([install guide](https://www.anaconda.com/docs/getting-started/miniconda/install)) | What Miniconda is and the official installer steps for each system |
| conda documentation, *Getting started with conda* ([docs.conda.io](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html)) | The whole page | The first conda commands and the idea of environments |
| conda documentation, *Managing environments* ([docs.conda.io](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)) | "Creating an environment", "Activating an environment", "Exporting an environment file", "Removing an environment" | Every environment command used in this topic |
| conda documentation, *Managing Python* ([docs.conda.io](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-python.html)) | The whole page | Choosing and changing the Python version in an environment |
| conda-forge, *A brief introduction* ([conda-forge.org](https://conda-forge.org/docs/user/introduction/)) | "How can I install packages from conda-forge?" | The free community channel, and Miniforge |

## Editor, notebooks, and Colab

| Source | Where to look | Covers |
|---|---|---|
| Visual Studio Code, *Getting started with Python in VS Code* ([code.visualstudio.com](https://code.visualstudio.com/docs/python/python-tutorial)) | "Install a Python interpreter" and "Create a virtual environment" | Setting up Python for the editor and creating an environment from inside it |
| Visual Studio Code, *Python environments in VS Code* ([code.visualstudio.com](https://code.visualstudio.com/docs/python/environments)) | "Select an environment" | Matching the editor to a `venv` or conda environment |
| Visual Studio Code, *Jupyter Notebooks in VS Code* ([code.visualstudio.com](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)) | "Select a kernel" and "Work with code cells" | Running notebooks in the editor |
| Visual Studio Code download page ([code.visualstudio.com](https://code.visualstudio.com/download)) | Installers for each system | Installing the editor |
| Project Jupyter, *Installing Jupyter* ([jupyter.org](https://jupyter.org/install)) | JupyterLab and Jupyter Notebook | Starting `jupyter lab` |
| IPython documentation, *Installing the IPython kernel* ([ipython.readthedocs.io](https://ipython.readthedocs.io/en/stable/install/kernel_install.html)) | "Kernels for different environments" | Registering an environment as a Jupyter kernel with `ipykernel install` |
| Google, *Colaboratory FAQ* ([research.google.com](https://research.google.com/colaboratory/faq.html)) | The questions on resource limits, installing libraries, and GPUs | What Colab provides, how long sessions last, and GPU use |
| Google Colab ([colab.research.google.com](https://colab.research.google.com/)) | The welcome notebook | Opening a notebook in the browser |

## Windows, git, and GPUs

| Source | Where to look | Covers |
|---|---|---|
| Microsoft, *Install WSL* ([learn.microsoft.com](https://learn.microsoft.com/en-us/windows/wsl/install)) | The whole page | Setting up a Linux environment inside Windows |
| Microsoft, *About Execution Policies* ([learn.microsoft.com](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies)) | The policy list and the scope options | Why PowerShell blocks scripts and what `RemoteSigned` means |
| git, *Downloads* ([git-scm.com](https://git-scm.com/downloads)) and Chacon & Straub, *Pro Git*, 2nd ed. ([git-scm.com/book](https://git-scm.com/book/en/v2)) | Downloads page; *Pro Git* Ch. 1 (Getting Started) and Ch. 2.1 (Git Basics: Getting a Git Repository) | Installing git, and a free, thorough book if you want more |
| GitHub Docs, *Cloning a repository* ([docs.github.com](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)) | The "Cloning a repository" section | Cloning over HTTPS or SSH |
| PyTorch, *Get started locally* ([pytorch.org](https://pytorch.org/get-started/locally/)) | The install selector | The right install command for your system and GPU, for the deep-learning course |

## Try

- The [notebook](notebook.ipynb) in this topic, then the [exercises](exercises/problems.md) and the [assignment](assignment.md).
- Next topic: [Python Basics](../python-basics/README.md).
