import pandas as pd

import jax
from jax import numpy as jnp

from functools import partial
from sklearn import preprocessing

from .utils import dict_to_binary, get_expr, frac_nonzero, masked_max


def annotate_snap(
    adata,
    marker_dict,
    group_name,
    layer=None,
    auc_weight=0.5,
    expr_weight=0.5,
):
    """
    Annotate cell types based on AUROC and expression of predefined marker genes.

    Parameters
    ----------
    adata
        AnnData object
    marker_dict
        Dict with marker genes for each celltype
    group_name
        Name of the column in adata.obs that contains the cluster labels
    layer
        Layer in adata to use for expression
    """
    # Reformat marker_dict into binary matrix
    marker_mat = dict_to_binary(marker_dict)

    # Compute AUROC and fraction nonzero for marker features
    features = marker_mat.columns
    metrics = auc_expr(adata, group_name, features=features)

    marker_mat = marker_mat.loc[:, metrics["features"]]
    auc_max = masked_max(metrics["auroc"], marker_mat.values)
    expr_max = masked_max(metrics["frac_nonzero"], marker_mat.values)
    # Combine metrics
    assignment_scores = (auc_weight * auc_max + expr_weight * expr_max) / (
        auc_weight + expr_weight
    )
    assign_idx = jnp.argmax(assignment_scores, axis=0)
    # Mask out genes that are not expressed in any cell
    assign_class = marker_mat.index[assign_idx]

    assign_df = pd.DataFrame(
        {
            "class": assign_class,
            "score": assignment_scores[assign_idx, jnp.arange(auc_max.shape[1])],
            "auc": auc_max[assign_idx, jnp.arange(auc_max.shape[1])],
            "expr": expr_max[assign_idx, jnp.arange(expr_max.shape[1])],
        },
        index=metrics["groups"],
    )

    return assign_df


def auc_expr(adata, group_name, features=None, layer=None):
    """Computes AUROC and fraction nonzero for each gene in an adata object."""
    # Turn string groups into integers
    le = preprocessing.LabelEncoder()
    le.fit(adata.obs[group_name])

    # Compute AUROC and fraction nonzero
    groups = jnp.array(le.transform(adata.obs[group_name]))
    expr, features = get_expr(adata, features=features, layer=layer)
    auroc, frac_nonzero = expr_auroc_over_groups(expr, groups)

    return dict(
        frac_nonzero=frac_nonzero,
        auroc=auroc,
        features=features,
        groups=le.classes_,
    )


@jax.jit
@partial(jax.vmap, in_axes=[1, None])
def jit_auroc(x, groups):
    # TODO: compute frac nonzero here to avoid iterating twice

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
