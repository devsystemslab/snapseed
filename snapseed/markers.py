import pandas as pd

import jax
from jax import numpy as jnp

from functools import partial

from .auroc import auc_expr


def find_markers(adata, group_name, features=None, layer=None):
    """
    Find markers for each cluster.

    Parameters
    ----------
    adata
        AnnData object
    group_name
        Name of the column in adata.obs that contains the cluster labels
    features
        List of features to use for marker identification
    layer
        Layer in adata to use for expression

    Returns
    -------
    DataFrame with AUROC and detection ratio for each gene and cluster.
    """
    # Get expression matrix
    expr, features = get_expr(adata, features=features, layer=layer)
    # Compute AUROC and fraction nonzero for marker features
    metrics = auc_expr(adata, group_name, features=features)
    # Combine metrics
    marker_scores = metrics["auroc"] * metrics["frac_nonzero"]

    marker_df = pd.DataFrame(
        index=metrics["features"],
        columns=metrics["groups"],
    )
    return marker_dict
