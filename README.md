# Translation Invariance Predicts Length Generalization

Code, data, and manuscript for the paper:

> **Translation Invariance in Attention Predicts Length Generalization: A Controlled Study of Positional Encoding Schemes**
> Roopesh G A, VIT Vellore

## Key Finding

A translation-invariance score (TI) — the R² of a linear fit between attention weight and query-key offset — correlates with extrapolation accuracy at r=0.713 (p=9.2×10⁻⁵) across 24 controlled experiments.

## Repository Structure

```
├── latex/
│   ├── manuscript.tex          # LaTeX source (IEEE two-column)
│   ├── manuscript.pdf          # Compiled 9-page paper
│   ├── references.bib          # 21 verified references with DOIs
│   └── compilation_notes.md    # Overleaf instructions
├── figures/
│   ├── fig1_accuracy_curves.*  # Per-length accuracy for all PE schemes
│   ├── fig2_invariance_scatter.* # TI vs extrapolation scatter plot
│   └── fig3_attention.*        # Attention heatmaps (ALiBi vs APE)
├── code/
│   ├── lengen.py               # TinyGPT model with selectable PE
│   ├── train_lengen.py         # Training and evaluation runner
│   ├── scan_main.py            # Orchestrator for all 24 experiments
│   ├── analyze.py              # Results analysis and plotting
│   └── attn_fig.py             # Attention visualization
├── results/                    # Raw JSON from all 24 runs
├── summary.csv                 # Aggregated results table
└── lengen_experiments.ipynb    # Self-contained Colab notebook
```

## Reproducing the Experiments

### Google Colab (recommended)
Upload `lengen_experiments.ipynb` to Colab with a T4 GPU runtime. All 24 experiments complete in ~25 minutes.

### Local
```bash
pip install torch
cd code
python scan_main.py
python analyze.py
```

## Experimental Setup

- **Model**: TinyGPT — 2-layer, 4-head decoder, d=64, 38k parameters
- **Tasks**: digit-string copy and reversal
- **PE schemes**: APE (learned absolute), NoPE (none), ALiBi, RoPE
- **Training**: lengths 1-10, tested up to length 18
- **Seeds**: 3 per configuration (24 runs total)

## Citation

```bibtex
@article{ga2026translation,
  title={Translation Invariance in Attention Predicts Length Generalization:
         A Controlled Study of Positional Encoding Schemes},
  author={G A, Roopesh},
  year={2026}
}
```

## License

MIT
