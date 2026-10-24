# CLAUDE.md

Rules for anyone (human or AI) writing content in this repo.

## Audience and tone
- Complete beginners with high-school math. Intuition first, formalism second.
- Friendly, clear, encouraging. No unexplained jargon.

## Hard rules
- Only original text, code, and figures written for this project.
- Never add books, slides, syllabi, papers, or other third-party material (no PDF, PPTX, DOCX, EPUB, copied images, or screenshots).
- Never copy or closely paraphrase text from sources. Write in your own words; draw figures yourself.
- `resources.md` files hold titles, authors, chapter pointers, and free, legitimate links only.
- No grading, attendance, scholarship, exam, or other institutional policies, and do not name any institution whose syllabi informed this project.

## Naming
- Kebab-case topic folders. No week numbers, no numeric prefixes.
- Study order lives in each course README, not in folder names.

## Topic template
- Required: `README.md` (learning goals, prerequisites, outline, key intuition, "Why this matters for AI/ML", further reading, Previous | Next links) and `resources.md`.
- Optional, only where they add real value: `notebook.ipynb` (with an Open in Colab badge), `exercises/problems.md` + `exercises/solutions.md`, `assignment.md`, `paper.md` (deep learning: the seminal paper in our own words), `lab.md` (build from scratch first, then use the library), `assets/` (only figures we generated).

## Content rules
- Connect every topic to a concrete AI/ML use case and link back to earlier math.
- Concept ownership: the first topic where a concept appears teaches it fully. Later topics recap briefly, link back, and cover only what is new. See the ownership map in `docs/design-notes.md`.
- Keep the topic order from the study-order lists. Ask before reordering.
- Python code must run top to bottom. Keep math and ML notebooks CPU-friendly; mark GPU-heavy deep-learning notebooks "GPU recommended". Commit notebooks executed with small outputs.

## Documentation
- GitHub-native Markdown only. Math as `$...$` and `$$...$$`. Diagrams as Mermaid or our own PNG/SVG in `assets/`.
- Relative links only for internal pages.
- New topic: `python scripts/new_topic.py <course> <topic-name>`, then add the printed line to the course README.
- Before every commit: run `python scripts/check_repo.py` and run any changed notebooks.
