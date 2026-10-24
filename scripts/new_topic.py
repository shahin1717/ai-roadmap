#!/usr/bin/env python3
"""Scaffold a topic folder from the template.

Usage:
    python scripts/new_topic.py <course> <topic-name> [--title TITLE]
        [--notebook] [--exercises] [--assignment] [--paper] [--lab]

<course> is the course folder relative to the repo root, for example
`machine-learning` or `math/linear-algebra`. Only README.md and resources.md
are always created; the optional files are added only where they add value.
Prints the line to add to the course README's study order.
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "shahin1717/ai-roadmap"
ACRONYMS = {"knn": "KNN", "mle": "MLE", "kkt": "KKT", "lu": "LU", "gmm": "GMM", "gpt": "GPT",
            "llms": "LLMs", "mlops": "MLOps", "cnn": "CNN", "ai": "AI", "ml": "ML", "numpy": "NumPy"}
SMALL = {"and", "of", "to", "the", "a", "with", "its", "this"}


def make_title(slug):
    words = slug.split("-")
    return " ".join(ACRONYMS.get(w) or (w if i and w in SMALL else w.capitalize())
                    for i, w in enumerate(words))


def readme(title, course_title, prev_link, next_link):
    return f"""# {title}

> **Status:** planned. This topic has not been written yet.

**Course:** [{course_title}](../README.md)

## Learning goals

## Prerequisites

## Outline

## Key intuition

## Why this matters for AI/ML

## Further reading

See [resources.md](resources.md).

---

{prev_link} | {next_link}
"""


def notebook(path, title):
    url = f"https://colab.research.google.com/github/{REPO_URL}/blob/main/{path}/notebook.ipynb"
    badge = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"
    return json.dumps({
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": [f"{badge}\n\n# {title}"]}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5}, indent=1) + "\n"


def scaffold(course, slug, title=None, prev_link=None, next_link=None, notebook_=False,
             exercises=False, assignment=False, paper=False, lab=False):
    if re.match(r"^\d", slug) or not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", slug):
        raise SystemExit(f"topic name must be kebab-case without a numeric prefix: {slug}")
    course_dir = ROOT / course
    course_title = make_title(Path(course).name)
    title = title or make_title(slug)
    topic = course_dir / slug
    topic.mkdir(parents=True, exist_ok=True)
    back = "[Back to course](../README.md)"
    files = {
        "README.md": readme(title, course_title, prev_link or back, next_link or back),
        "resources.md": f"# Resources: {title}\n\nTitles, authors, chapter pointers, and free links only.\n",
    }
    if notebook_:
        files["notebook.ipynb"] = notebook(f"{course}/{slug}", title)
    if exercises:
        files["exercises/problems.md"] = f"# Exercises: {title}\n"
        files["exercises/solutions.md"] = f"# Solutions: {title}\n"
    if assignment:
        files["assignment.md"] = f"# Assignment: {title}\n"
    if paper:
        files["paper.md"] = f"# Paper of the topic: {title}\n"
    if lab:
        files["lab.md"] = f"# Lab: {title}\n"
    for name, text in files.items():
        f = topic / name
        f.parent.mkdir(parents=True, exist_ok=True)
        if not f.exists():
            f.write_text(text, encoding="utf-8")
    return f"[{title}]({slug}/README.md)"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("course")
    p.add_argument("topic")
    p.add_argument("--title")
    for flag in ("notebook", "exercises", "assignment", "paper", "lab"):
        p.add_argument(f"--{flag}", action="store_true")
    a = p.parse_args()
    line = scaffold(a.course, a.topic, a.title, notebook_=a.notebook, exercises=a.exercises,
                    assignment=a.assignment, paper=a.paper, lab=a.lab)
    print(f"Add to {a.course}/README.md study order:\n1. {line}")


if __name__ == "__main__":
    main()
