# arXiv draft

Submission-ready LaTeX for the POISE characterization.

## Files
- `main.tex` — the paper (self-contained; no external `.sty`, embedded bibliography).
- `figs/*.pdf` — vector figures, generated from the measured result JSONs.
- `make_figures.py` — regenerates every figure from `data/results/` (reproducible).
- `paper.md` — the same content in Markdown, for quick reading / GitHub.

## Build
```bash
# regenerate figures from the committed measurements (optional; PDFs are checked in)
python3.10 docs/arxiv/make_figures.py

# compile (needs a TeX distribution — not installed on the Jetson)
cd docs/arxiv && pdflatex main.tex && pdflatex main.tex   # twice for refs
```
No local TeX? Upload `main.tex` + the `figs/` folder to **Overleaf** or **arXiv** — both
compile it as-is.

## Submit to arXiv
1. Fill in the author name/affiliation in `main.tex` (`[Author Name]`).
2. Bundle `main.tex` + `figs/` (arXiv runs `pdflatex` server-side).
3. Suggested category: `cs.LG` (cross-list `cs.AR` / `cs.PF`).

Every number in the paper is an eval-harness output on the real device; figures derive only
from files under `data/results/`.
