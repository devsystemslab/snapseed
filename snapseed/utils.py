import jax
from jax import numpy as jnp

from functools import partial

from sklearn import preprocessing
import scanpy as sc

from sklearn.metrics import roc_auc_score

to_dense = lambda x: x.toarray() if hasattr(x, "toarray") else x


@jax.jit
@partial(jax.vmap, in_axes=[None, 0])
def masked_max(x, mask):
    return jnp.max(x * mask, axis=1)


@jax.jit
def frac_nonzero(x, axis=0):
    return jnp.mean(x > 0, axis=axis)


@jax.jit
@partial(jax.vmap, in_axes=[1, None])
def jit_auroc(x, groups):
    # sort scores and corresponding truth values
    desc_score_indices = jnp.argsort(x)[::-1]
    x = x[desc_score_indices]
    groups = groups[desc_score_indices]
    # x typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = jnp.array(jnp.diff(x) != 0, dtype=jnp.int32)
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
    auroc = jnp.zeros((groups.max() + 1, expr.shape[1]))
    frac_nz = jnp.zeros((groups.max() + 1, expr.shape[1]))
    for group in range(groups.max() + 1):
        auroc = auroc.at[group, :].set(jit_auroc(expr, groups == group))
        frac_nz = frac_nz.at[group, :].set(frac_nonzero(expr[groups == group, :]))
    return auroc, frac_nz


def auc_expr(adata, group_name, features=None, layer=None):
    """Computes AUROC and fraction nonzero for each gene in an adata object."""
    # Turn string groups into integers
    le = preprocessing.LabelEncoder()
    le.fit(adata.obs[group_name])
    # Compute AUROC and fraction nonzero
    groups = jnp.array(le.transform(adata.obs[group_name]))
    expr = get_expr(adata, features=features, layer=layer)
    auroc, frac_nonzero = expr_auroc_over_groups(expr, groups)
    return dict(
        frac_nonzero=frac_nonzero,
        auroc=auroc,
        features=features,
        groups=le.classes_,
    )


def dict_to_binary(d):
    df = pd.concat(
        [pd.Series(v, name=k).astype(str) for k, v in marker_dict.items()],
        axis=1,
    )
    marker_mat = pd.get_dummies(df.stack()).groupby(level=1).sum().clip(upper=1)
    return marker_mat


def get_expr(adata, features=None, layer=None):
    if features is not None:
        # intersect with adata features
        features = list(set(features) & set(adata.var_names))
    adata = adata[:, features] if features is not None else adata
    if layer is not None:
        expr = jnp.array(to_dense(adata[:, features].layers[layer]))
    else:
        expr = jnp.array(to_dense(adata[:, features].X))
    return expr, features
