# C5 best-match permutation audit

Run `python ../C5_bestmatch_permutation.py` from this directory, or invoke the
script from anywhere. It reads the 334 WordNet-matched concepts from `../lib/`,
then writes `C5_bestmatch_permutation.npz` here. The default is 2,000 row
permutations with `numpy.random.default_rng(20261004)`.

The observed statistic is the **largest positive Pearson correlation** among
20 CATS dimensions, separately for each of 30 model instances and each of 49
SPOSE dimensions. For every null draw, the script permutes the 334 rows of the
*entire* SPOSE matrix with one shared permutation, then repeats the same
20-way maximum. This preserves each dataset's feature values and within-dataset
correlations. It tests whether the observed alignment depends on matching the
same concept labels across the two datasets. It does not test a null in which
the feature matrices themselves are random, nor does it assert that concepts
are independent samples from a population.

The NPZ contains the observed best-match scores and winning CATS dimensions,
instance-level permutation p-values, 30-instance mean p-values, and max-over-49
FWER-corrected mean p-values. All p-values use `(exceedances + 1) / (N + 1)`.
The 0.107 circle is stored as `ring_reference`, a single-pair Pearson reference
line, **not** a best-match significance threshold. `null_single_q95` and
`null_ring_exceedance` quantify this distinction. `null_max_mean` records the
FWER null distribution; the complete `(2000, 30, 49)` null cube is regenerated
by the script rather than stored.
