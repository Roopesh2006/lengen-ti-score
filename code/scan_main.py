"""Definitive run: tasks x positional schemes x seeds, with per-length accuracy
and attention translation-invariance. ~24 runs, saved to results/."""
import itertools
from train_lengen import run

TASKS = ["copy", "reverse"]
PES = ["ape", "nope", "alibi", "rope"]
SEEDS = [0, 1, 2]
for task, pe, seed in itertools.product(TASKS, PES, SEEDS):
    run(pe=pe, seed=seed, steps=3000, train_hi=10, test_hi=18,
        tag="main", task=task)
