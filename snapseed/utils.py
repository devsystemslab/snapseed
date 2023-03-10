import jax
from jax import numpy as jnp

from functools import partial

from sklearn import preprocessing
import scanpy as sc

from sklearn.metrics import roc_auc_score

to_dense = lambda x: x.toarray() if hasattr(x, "toarray") else x


@jax.jit
def frac_nonzero(x, axis=0):
    return jnp.mean(x > 0, axis=axis)


@jax.jit
@partial(jax.vmap, in_axes=[1, None])
def jit_auroc(expr, groups):
    # sort scores and corresponding truth values
    desc_score_indices = jnp.argsort(expr)[::-1]
    expr = expr[desc_score_indices]
    groups = groups[desc_score_indices]
    # expr typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = jnp.array(jnp.diff(expr) != 0, dtype=jnp.int32)
    threshold_mask = jnp.r_[distinct_value_indices, 1]
    # accumulate the true positives with decreasing threshold
    tps_ = jnp.cumsum(groups)
    fps_ = 1 + jnp.arange(groups.size) - tps_
    # mask out the values that are not distinct
    tps = jnp.sort(tps_ * threshold_mask)
    fps = jnp.sort(fps_ * threshold_mask)
    tps = jnp.r_[0, tps]
    fps = jnp.r_[0, fps]
    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    area = jnp.trapz(tpr, fpr)
    return area


def expr_auroc_over_groups(expr, groups):
    """Computes AUROC for each group separately."""
    auroc = jnp.zeros((expr.shape[1], groups.max() + 1))
    frac_nz = jnp.zeros((expr.shape[1], groups.max() + 1))
    for group in range(groups.max() + 1):
        auroc = auroc.at[:, group].set(jit_auroc(expr, groups == group))
        frac_nz = frac_nz.at[:, group].set(frac_nonzero(expr[groups == group, :]))
    return auroc, frac_nz


def auc_expr(adata, group_name, features=None):
    """Computes AUROC and fraction nonzero for each gene in an AnnData object."""
    # Turn string groups into integers
    le = preprocessing.LabelEncoder()
    le.fit(adata.obs[group_name])
    # Compute AUROC and fraction nonzero
    groups = jnp.array(le.transform(adata.obs[group_name]))
    # Select features
    if features is not None:
        expr = jnp.array(to_dense(adata[:, features].X))
    else:
        expr = jnp.array(to_dense(adata.X))
    auroc, frac_nonzero = expr_auroc_over_groups(expr, groups)
    return dict(frac_nonzero=frac_nonzero, auroc=auroc)
