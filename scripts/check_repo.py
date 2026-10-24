#!/usr/bin/env python3
"""Validate the repo structure. Exits with status 1 if any check fails.

Checks:
- no third-party document files (.pdf, .docx, .pptx, .epub)
- every topic folder has README.md and resources.md
- no folder name starts with a number
- every topic is linked from its course README's study order
- every relative link in Markdown files and notebook Markdown cells resolves
- no bare `\\\\` (matrix row break) in math: GitHub's Markdown step eats one backslash,
  so use a ```math fence for display math or $`...`$ for inline math instead
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COURSES = ["getting-started", "math/linear-algebra", "math/calculus-and-optimization",
           "math/probability-and-statistics", "machine-learning", "deep-learning"]
REQUIRED = ["README.md", "resources.md"]
FORBIDDEN = {".pdf", ".docx", ".pptx", ".epub"}
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
SAFE_MATH = re.compile(r"```math\n.*?\n```|\$`.*?`\$", re.S)


def tracked_files():
    return [p for p in ROOT.rglob("*") if ".git" not in p.relative_to(ROOT).parts]


def markdown_sources(path):
    if path.suffix == ".md":
        return path.read_text(encoding="utf-8")
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def main():
    errors = []
    files = tracked_files()
    for p in files:
        rel = p.relative_to(ROOT)
        if p.is_file() and p.suffix.lower() in FORBIDDEN:
            errors.append(f"forbidden file type: {rel}")
        if p.is_dir() and re.match(r"^\d", p.name):
            errors.append(f"numeric prefix in folder name: {rel}")

    for course in COURSES:
        course_dir = ROOT / course
        readme = course_dir / "README.md"
        if not readme.exists():
            errors.append(f"missing course README: {course}/README.md")
            continue
        text = readme.read_text(encoding="utf-8")
        for topic in sorted(d for d in course_dir.iterdir() if d.is_dir()):
            for name in REQUIRED:
                if not (topic / name).exists():
                    errors.append(f"missing {name}: {topic.relative_to(ROOT)}")
            if f"]({topic.name}/" not in text:
                errors.append(f"not in study order of {course}/README.md: {topic.name}")

    for p in files:
        if p.is_file() and p.suffix in {".md", ".ipynb"}:
            for n, line in enumerate(SAFE_MATH.sub("", markdown_sources(p)).splitlines(), 1):
                if "\\\\" in line:
                    errors.append(f"bare \\\\ in math (use a ```math fence or $`...`$): {p.relative_to(ROOT)}:{n}")
            for target in LINK.findall(markdown_sources(p)):
                if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                    continue
                path = target.split("#")[0]
                if path and not (p.parent / path).exists():
                    errors.append(f"broken link in {p.relative_to(ROOT)}: {target}")

    for e in errors:
        print(f"FAIL  {e}")
    print(f"{len(errors)} problem(s) found." if errors else "All checks passed.")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
