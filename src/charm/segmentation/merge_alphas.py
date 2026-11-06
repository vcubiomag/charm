import numpy as np

from charm.logging import logger


def _check_probabilities(alphas, name):
    if np.max(np.abs(np.sum(alphas, axis=1) - 1)) > 1e-5:
        raise ValueError(f"{name} invalid: class probabilities in the mesh should sum to one in all nodes")


def merging_fractions_table(names, merge_options):
    """A classes x structures matrix where each column holds the fractions of the
    classes (super-structures) in that structure. Each column sums to 1."""
    fractions = np.array(
        [
            [any(search_string in name for search_string in option.search_strings) for name in names]
            for option in merge_options
        ],
        dtype=float,
    )
    if not fractions.any(axis=0).all():
        raise ValueError("some structures are not associated with any super-structures")
    fractions /= fractions.sum(axis=0)

    width = max(map(len, names))
    for option, class_fractions in zip(merge_options, fractions):
        logger.debug(option.merged_name.strip())
        for name, fraction in zip(names, class_fractions):
            if fraction > 0:
                logger.debug(f"    {name.ljust(width)} ({int(fraction * 100)}%)")

    return fractions


def merge_alphas(alphas, fractions):
    """Add together the columns of alphas that belong to the same super-structure."""
    _check_probabilities(alphas, "alphas")
    merged = alphas @ fractions.T
    _check_probabilities(merged, "merged alphas")
    return merged
