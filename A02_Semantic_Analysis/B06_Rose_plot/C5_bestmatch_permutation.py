"""Conditional permutation audit of the C3/C4 best-match rose-chart statistic.

Inputs are the same 334 WordNet-matched concepts used by C0/C3: 30 CATS
trials (21..50), 20 dimensions each, and the 49 THINGS/SPOSE dimensions.
Each permutation applies ONE common row permutation to the entire SPOSE
matrix, preserving both datasets' internal multivariate structure. The
20-dimensional maximum is recomputed for every trial and SPOSE dimension.

Run: python C5_bestmatch_permutation.py [--permutations 2000]
Output: results/C5_bestmatch_permutation.npz
Requires: numpy, scipy. The default_rng seed/API and float32 null storage
match the 2026-10-04 audit, so its reported numbers can be reproduced.
"""

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import t


HERE = Path(__file__).resolve().parent
SEED = 20261004
EXAMPLES = (0, 1, 4, 19)  # metal/tool, food, furniture, long/thin


def load_inputs():
    lib = HERE / "lib"
    matches = json.loads((lib / "wordnet_matches.json").read_text(encoding="utf-8"))["matches"]
    rows = list(matches.values())  # C0 preserves JSON insertion order
    imagenet_idx = np.array([row["imagenet_index"] for row in rows])
    things_idx = np.array([row["things_index"] for row in rows])
    if len(rows) != 334:
        raise ValueError(f"Expected 334 matched concepts, found {len(rows)}")

    cats = []
    for trial in range(21, 51):
        path = lib / "Concept_Vec" / f"symbol_set_trail{trial}.pt"
        # These files store a single uncompressed 1000x20 float32 tensor.
        # Reading the tensor payload avoids a dependency on torch at audit time.
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.endswith("/data/0")]
            if len(members) != 1:
                raise ValueError(f"Unexpected tensor layout: {path}")
            values = np.frombuffer(archive.read(members[0]), dtype="<f4")
        if values.size != 1000 * 20:
            raise ValueError(f"Unexpected tensor shape: {path}")
        cats.append(values.reshape(1000, 20)[imagenet_idx])

    cats = np.stack(cats).astype(np.float64)
    spose = np.loadtxt(lib / "THINGS_Features.txt", dtype=np.float64)[things_idx]
    if cats.shape != (30, 334, 20) or spose.shape != (334, 49):
        raise ValueError(f"Unexpected shapes: {cats.shape}, {spose.shape}")
    return cats, spose


def unit_center(values, axis):
    centered = values - values.mean(axis=axis, keepdims=True)
    norm = np.linalg.norm(centered, axis=axis, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("A constant feature cannot have a Pearson correlation")
    return centered / norm


def bh_adjust(p):
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "C5_bestmatch_permutation.npz")
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("--permutations must be positive")

    cats, spose = load_inputs()
    cats = unit_center(cats, axis=1)
    spose = unit_center(spose, axis=0)
    observed_r = np.einsum("mnd,nk->mdk", cats, spose, optimize=True)
    observed = observed_r.max(axis=1)
    winner = observed_r.argmax(axis=1)
    observed_mean = observed.mean(axis=0)
    flat = cats.transpose(0, 2, 1).reshape(600, 334)

    rng = np.random.default_rng(SEED)
    null = np.empty((args.permutations, 30, 49), dtype=np.float32)
    for index in range(args.permutations):
        permutation = rng.permutation(334)
        correlations = (flat @ spose[permutation]).reshape(30, 20, 49)
        null[index] = correlations.max(axis=1)
        if (index + 1) % 500 == 0:
            print(f"Permutations: {index + 1}/{args.permutations}", flush=True)

    count = args.permutations + 1  # plus-one Monte Carlo p-values
    p_instance = ((null >= observed[None]).sum(axis=0) + 1) / count
    null_mean = null.mean(axis=1)
    p_mean = ((null_mean >= observed_mean[None]).sum(axis=0) + 1) / count
    null_max_mean = null_mean.max(axis=1)
    p_mean_fwer = ((null_max_mean[:, None] >= observed_mean[None]).sum(axis=0) + 1) / count
    q_mean_bh = bh_adjust(p_mean)
    df = 334 - 2
    tcrit = t.ppf(0.975, df)
    single_pair_rcrit = tcrit / np.sqrt(tcrit**2 + df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        seed=SEED, permutations=args.permutations,
        imagenet_trials=np.arange(21, 51), example_indices=np.array(EXAMPLES),
        observed_bestmatch=observed, winning_cats_dimension=winner,
        observed_mean=observed_mean, p_instance=p_instance, p_mean=p_mean,
        p_mean_fwer=p_mean_fwer, q_mean_bh=q_mean_bh,
        null_max_mean=null_max_mean, null_single_q95=np.quantile(null, 0.95),
        null_ring_exceedance=(null > 0.107).mean(),
        null_mean_q95=np.quantile(null_mean, 0.95, axis=0),
        single_pair_rcrit=single_pair_rcrit, ring_reference=0.107,
    )
    print(f"Output: {args.output}")
    print(f"Single-pair rcrit: {single_pair_rcrit:.6f}; null best-match q95: {np.quantile(null, .95):.6f}")
    print(f"Null fraction above 0.107 ring: {(null > .107).mean():.6f}")
    for feature in EXAMPLES:
        print(f"Feature {feature}: mean={observed_mean[feature]:.6f}, "
              f"above ring={int((observed[:, feature] > .107).sum())}/30, "
              f"instance p<.05={int((p_instance[:, feature] < .05).sum())}/30, "
              f"winner dimensions={len(np.unique(winner[:, feature]))}, "
              f"FWER p={p_mean_fwer[feature]:.6f}")
    print(f"FWER-significant SPOSE dimensions: {(p_mean_fwer < .05).sum()}/49")


if __name__ == "__main__":
    main()
