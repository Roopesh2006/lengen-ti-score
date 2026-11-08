"""Aggregate results/*.json into tables + publication figures for the paper."""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
PES = ["ape", "rope", "nope", "alibi"]
LABELS = {"ape": "Learned APE", "nope": "NoPE", "alibi": "ALiBi", "rope": "RoPE"}
COL = {"ape": "#d1495b", "rope": "#edae49", "nope": "#66a182", "alibi": "#2e4057"}
TASKS = ["copy", "reverse"]


def load(task, pe, tag="main"):
    rows = []
    for fn in glob.glob(os.path.join(RES, f"lengen_{task}_{pe}_{tag}_s*.json")):
        with open(fn) as f:
            rows.append(json.load(f))
    return rows


def curve(rows):
    """mean/std per-length accuracy across seeds."""
    Ls = sorted(int(k) for k in rows[0]["accs"].keys())
    M = np.array([[r["accs"][str(L)] for L in Ls] for r in rows])
    return np.array(Ls), M.mean(0), M.std(0)


def main():
    any_rows = None
    # ---- Figure 1: per-length accuracy, one panel per task ----
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    train_hi = None
    for ax, task in zip(axes, TASKS):
        for pe in PES:
            rows = load(task, pe)
            if not rows:
                continue
            any_rows = rows
            train_hi = rows[0]["train_hi"]
            Ls, mu, sd = curve(rows)
            ax.plot(Ls, mu, "-o", ms=3, lw=1.4, color=COL[pe], label=LABELS[pe])
            ax.fill_between(Ls, mu - sd, mu + sd, color=COL[pe], alpha=0.15, lw=0)
        ax.axvspan(0.5, train_hi + 0.5, color="0.9", zorder=0)
        ax.axvline(train_hi + 0.5, color="0.4", ls="--", lw=0.8)
        ax.set_title(f"{task}", fontsize=10)
        ax.set_xlabel("sequence length (digits)")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("exact-match accuracy")
    axes[0].text(train_hi/2, 1.06, "train", ha="center", fontsize=7, color="0.4")
    axes[0].text(train_hi+4, 1.06, "extrapolation", ha="center", fontsize=7, color="0.4")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_lengen_curves.pdf"))
    fig.savefig(os.path.join(FIG, "fig1_lengen_curves.png"))
    plt.close(fig)

    # ---- Figure 2: invariance score vs extrapolation accuracy (the predictor) ----
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    xs, ys = [], []
    for task in TASKS:
        for pe in PES:
            rows = load(task, pe)
            if not rows:
                continue
            th = rows[0]["train_hi"]; tt = rows[0]["test_hi"]
            for r in rows:
                inv = r["invariance"]
                ex = np.mean([r["accs"][str(L)] for L in range(th + 1, tt + 1)])
                xs.append(inv); ys.append(ex)
                mk = "o" if task == "copy" else "s"
                ax.scatter(inv, ex, c=COL[pe], marker=mk, s=28, edgecolor="w", lw=0.4, zorder=3)
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) > 2:
        r = np.corrcoef(xs, ys)[0, 1]
        b, a = np.polyfit(xs, ys, 1)
        xg = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(xg, a + b * xg, "k--", lw=1, zorder=2,
                label=f"r = {r:.2f}")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_xlabel("attention translation-invariance  (offset $R^2$)")
    ax.set_ylabel("extrapolation accuracy")
    ax.set_ylim(-0.03, 1.03)
    # legend proxies for schemes
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker="o", color="w", markerfacecolor=COL[p],
               markersize=7, label=LABELS[p]) for p in PES]
    leg2 = ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="lower right",
                     title="scheme", title_fontsize=7.5)
    ax.add_artist(leg2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_invariance_predictor.pdf"))
    fig.savefig(os.path.join(FIG, "fig2_invariance_predictor.png"))
    plt.close(fig)

    # ---- summary table (printed + saved) ----
    lines = ["task,scheme,train_acc,extrap_acc,invariance"]
    summary = {}
    for task in TASKS:
        for pe in PES:
            rows = load(task, pe)
            if not rows:
                continue
            th = rows[0]["train_hi"]; tt = rows[0]["test_hi"]
            tr = np.mean([[r["accs"][str(L)] for L in range(1, th+1)] for r in rows])
            exs = [np.mean([r["accs"][str(L)] for L in range(th+1, tt+1)]) for r in rows]
            invs = [r["invariance"] for r in rows]
            lines.append(f"{task},{pe},{tr:.3f},{np.mean(exs):.3f}±{np.std(exs):.3f},"
                         f"{np.mean(invs):.3f}±{np.std(invs):.3f}")
            summary[(task, pe)] = dict(train=tr, extrap=np.mean(exs),
                                       extrap_sd=np.std(exs), inv=np.mean(invs))
    out = "\n".join(lines)
    with open(os.path.join(RES, "summary.csv"), "w") as f:
        f.write(out + "\n")
    print(out)
    print("\ncorrelation(invariance, extrap) =", np.corrcoef(xs, ys)[0, 1] if len(xs)>2 else "NA")
    return summary


if __name__ == "__main__":
    main()
