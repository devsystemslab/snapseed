import pandas as pd

import jax
from jax import numpy as jnp

from .utils import auc_expr, dict_to_binary


def annotate(adata, marker_dict, group_name, method="snap"):
    """Annotate clusters with marker genes."""
    if method == "snap":
        assignments = annotate_snap(adata, marker_dict, group_name)
    if method == "cytograph":
        raise NotImplementedError("Cytograph annotation not yet implemented.")
        # assignments = annotate_cytograph(adata, marker_dict, group_name)
    # Join cluster-level results with adata
    assignments = assignments.reset_index(names=group_name)
    return assignments


def annotate_snap(adata, marker_dict, group_name):
    """
    Anntoate cell types based on AUROC and expression of predefined marker genes.

    Parameters
    ----------
    adata
        AnnData object
    marker_dict
        Dict with marker genes for each celltype
    group_name
        Name of the column in adata.obs that contains the cluster labels
    """
    # Reformat marker_dict into binary matrix
    marker_mat = dict_to_binary(marker_dict)
    # Compute AUROC and fraction nonzero for marker features
    features = marker_mat.columns
    metrics = auc_expr(adata, group_name, features=features)
    # Subset markers to actually used features
    marker_mat = marker_mat.loc[:, metrics["features"]]
    auc_max = masked_max(metrics["auroc"], marker_mat.values)
    expr_max = masked_max(metrics["frac_nonzero"], marker_mat.values)
    assignment_scores = auc_max * expr_max
    assign_idx = jnp.argmax(assignment_scores, axis=0)
    assign_df = pd.DataFrame(
        {
            "class": marker_mat.index[assign_idx],
            "score": assignment_scores[assign_idx, jnp.arange(auc_max.shape[1])],
            "auc": auc_max[assign_idx, jnp.arange(auc_max.shape[1])],
            "expr": expr_max[assign_idx, jnp.arange(expr_max.shape[1])],
        },
        index=metrics["groups"],
    )
    return assign_df


annotate(adata, marker_dict, "res.0.6")

mask = marker_mat.values[0, :]
x = metrics["auroc"]

x * mask

marker_dict = dict(
    Dorsal_Telencephalon=["EMX1", "NEUROD6", "NFIX"],
    Ventral_Telencephalon=["DLX5", "DLX2"],
    Diencephalon=["NHLH2", "LHX5", "RSPO3", "RSPO2"],
    Mesencephalon=["OTX2", "LHX1", "LHX5", "ZIC1"],
    Rhombencephalon=["HOXB2"],
)

group_name = "res.0.6"
adata = adata

auc_expr(adata, "res.0.6")

# Jax 2d array
# Test groups
groups = jnp.array([0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
# Expr mat with same nrows as groups
expr_mat = jnp.array(
    [
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.3],
        [0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.3],
        [0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.3],
        [0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.3],
        [0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3],
    ]
)
