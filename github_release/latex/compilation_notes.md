# Compilation Notes

## Files
- `manuscript.tex` — main LaTeX source
- `references.bib` — BibTeX database (21 verified entries)
- `fig1_accuracy_curves.pdf` — per-length accuracy curves (Figure 1)
- `fig2_invariance_scatter.pdf` — TI score vs extrapolation scatter (Figure 2)
- `fig3_attention.pdf` — attention heatmaps (Figure 3)

## Compilation (local)
```bash
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

## Overleaf
1. Upload all files to a new Overleaf project.
2. **Important**: change the document class back to IEEE format by replacing the first line:
   ```latex
   % Replace this:
   \documentclass[10pt,twocolumn]{article}
   % With this:
   \documentclass[journal]{IEEEtran}
   ```
3. Remove the `\usepackage[margin=0.75in]{geometry}`, `\usepackage{abstract}`, and the two `\renewcommand` lines that follow it.
4. Restore author block to IEEE format:
   ```latex
   \author{Roopesh~G~A\thanks{...}}
   \markboth{IEEE Transactions, 2026}{...}
   ```
5. Restore `\begin{IEEEkeywords}...\end{IEEEkeywords}`.
6. Change `\bibliographystyle{ieeetr}` to `\bibliographystyle{IEEEtran}`.
7. Overleaf has IEEEtran.cls built in — it will compile immediately.

## Target Journal Suggestions
- **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)** — Impact Factor ~24
- **Transactions on Machine Learning Research (TMLR)** — open-access, high visibility
- **NeurIPS / ICML / ICLR** workshops — conference track for preliminary results

## Paper Statistics
- Word count: ~4,500 (main text)
- References: 21
- Figures: 3
- Tables: 1
- Pages: 7 (two-column)
- Experimental runs: 24 (4 PEs × 2 tasks × 3 seeds)

## Areas to Manually Verify
- GitHub repository URL (placeholder: https://github.com/roopeshga/lengen-ti-score)
- Co-author details (to be added by user)
- GPU compute time estimate (~25 minutes on T4)
