"""Train short APE and ALiBi copy models; dump averaged attention maps at a
training length and an extrapolation length to visualise absolute vs relative
structure. Produces figures/fig3_attention.pdf."""
import os, time
import numpy as np
import torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lengen import TinyGPT, make_task, encode_batch, V

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)


def train(pe, steps=2500, train_hi=10, Lmax=48, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = TinyGPT(d=64, nh=4, nl=2, pe=pe, Lmax=Lmax, ff=128)
    opt = torch.optim.Adam(m.parameters(), lr=5e-4)
    for s in range(steps):
        items = [make_task("copy", int(rng.integers(1, train_hi + 1)), rng) for _ in range(128)]
        x, tgt, mask = encode_batch(items, Lmax)
        logits, _ = m(x)
        loss = (F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="none")
                .reshape(tgt.shape) * mask).sum() / mask.sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


@torch.no_grad()
def avg_attn(model, n, Lmax, seed=1):
    rng = np.random.default_rng(seed)
    items = [make_task("copy", n, rng) for _ in range(64)]
    x, _, _ = encode_batch(items, Lmax)
    L0 = 2 * n + 2
    _, attn = model(x, need_attn=True)
    A = attn[0][0][:, :L0, :L0].mean(0).cpu().numpy()     # layer0 head0, avg over batch
    return A


def main():
    t0 = time.time()
    fig, axes = plt.subplots(2, 2, figsize=(5.4, 5.2))
    for col, pe in enumerate(["ape", "alibi"]):
        m = train(pe)
        for row, n in enumerate([10, 15]):     # train length, extrapolation length
            A = avg_attn(m, n, 48)
            ax = axes[row, col]
            im = ax.imshow(A, cmap="magma", aspect="equal")
            ax.set_title(f"{'Learned APE' if pe=='ape' else 'ALiBi'}  (len {n}"
                         + ("" if n == 10 else ", extrapolation") + ")", fontsize=8)
            ax.set_xlabel("key position"); ax.set_ylabel("query position")
            ax.tick_params(labelsize=6)
        print(f"{pe} done ({time.time()-t0:.0f}s)", flush=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_attention.pdf"))
    fig.savefig(os.path.join(FIG, "fig3_attention.png"))
    plt.close(fig)
    print("saved fig3", flush=True)


if __name__ == "__main__":
    main()
