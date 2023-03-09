import jax
from jax import numpy as jnp


def auroc(y_true, y_score):
    """Binary roc auc score."""
    if len(jnp.unique(y_true)) != 2:
        raise ValueError("AUROC is not defined for one group.")
    return jit_auroc(y_true, y_score)


@jax.jit
def jit_auroc(y_true, y_score):
    # make y_true a boolean vector
    y_true = y_true == 1
    # sort scores and corresponding truth values
    desc_score_indices = jnp.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    # accumulate the true positives with decreasing threshold
    tps = jnp.cumsum(y_true)
    fps = 1 + jnp.arange(len(y_true)) - tps
    tps = jnp.r_[0, tps]
    fps = jnp.r_[0, fps]
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    area = jnp.trapz(tpr, fpr)
    return area


auroc(y_true, y_score)
