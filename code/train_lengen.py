"""Train TinyGPT on addition; evaluate exact-match accuracy per operand length.
Saves per-length accuracy curves (train lengths + extrapolation) to results/."""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from lengen import (TinyGPT, make_example, make_task, encode_batch,
                    translation_invariance, V, PAD, STOI)

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)
VOCAB_INV = {v: k for k, v in STOI.items()}


def _item(task, n, rng):
    if task == "add":
        return make_example(n, n, rng)
    return make_task(task, n, rng)


def sample_batch(B, lo, hi, rng, Lmax, mixed=True, task="copy"):
    items = []
    for _ in range(B):
        n = rng.integers(lo, hi + 1) if mixed else hi
        items.append(_item(task, int(n), rng))
    return encode_batch(items, Lmax)


@torch.no_grad()
def eval_len(model, n, rng, Lmax, n_eval=400, task="copy"):
    model.eval()
    correct = 0
    items = [_item(task, n, rng) for _ in range(n_eval)]
    x, tgt, mask = encode_batch(items, Lmax)
    logits, _ = model(x)
    pred = logits.argmax(-1)
    # exact match over the masked (answer) region, per example
    m = mask.bool()
    for i in range(x.shape[0]):
        idx = m[i].nonzero().squeeze(-1)
        if len(idx) == 0:
            continue
        if torch.equal(pred[i, idx], tgt[i, idx]):
            correct += 1
    model.train()
    return correct / x.shape[0]


def run(pe="nope", seed=0, train_lo=1, train_hi=7, test_hi=14, d=64, nh=4, nl=2,
        Lmax=48, ff=128, steps=5000, B=128, lr=5e-4, mixed=True, tag="", task="copy"):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = TinyGPT(d=d, nh=nh, nl=nl, pe=pe, Lmax=Lmax, ff=ff)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 200) * (0.5 ** (s / max(1, steps))))
    t0 = time.time()
    for s in range(steps):
        x, tgt, mask = sample_batch(B, train_lo, train_hi, rng, Lmax, mixed=mixed, task=task)
        logits, _ = model(x)
        loss = (F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1),
                                reduction="none").reshape(tgt.shape) * mask).sum() / mask.sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    # per-length accuracy
    accs = {}
    for n in range(1, test_hi + 1):
        accs[n] = eval_len(model, n, np.random.default_rng(9000 + n), Lmax, task=task)
    # translation-invariance of learned attention (probe at a training length)
    prng = np.random.default_rng(5000 + seed)
    probe_items = [_item(task, train_hi, prng) for _ in range(64)]
    inv = translation_invariance(model, probe_items, Lmax)
    out = dict(pe=pe, seed=seed, task=task, train_lo=train_lo, train_hi=train_hi,
               test_hi=test_hi, mixed=mixed, steps=steps, d=d, nl=nl, nh=nh,
               accs=accs, invariance=inv,
               train_time=time.time() - t0, final_loss=float(loss.detach()))
    fn = os.path.join(RES, f"lengen_{task}_{pe}_{tag}_s{seed}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=2)
    tr = np.mean([accs[n] for n in range(train_lo, train_hi + 1)])
    ex = np.mean([accs[n] for n in range(train_hi + 1, test_hi + 1)])
    print(f"[{task[:3]} {pe:5s} {tag} s{seed}] train={tr:.3f} extrap={ex:.3f} "
          f"inv={inv:.3f} ({out['train_time']:.0f}s)  per-len="
          + " ".join(f"{accs[n]:.2f}" for n in range(1, test_hi + 1)), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe", default="nope")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--train_hi", type=int, default=7)
    ap.add_argument("--test_hi", type=int, default=14)
    ap.add_argument("--mixed", type=int, default=1)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        run(pe=a.pe, seed=0, steps=400, train_hi=5, test_hi=8, tag="smoke")
    else:
        run(pe=a.pe, seed=a.seed, steps=a.steps, train_hi=a.train_hi,
            test_hi=a.test_hi, mixed=bool(a.mixed), tag=a.tag)
