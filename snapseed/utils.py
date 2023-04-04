import yaml
import pandas as pd
import numba

import jax
from jax import numpy as jnp

from functools import partial
from sklearn.metrics import roc_auc_score

import scanpy as sc

to_dense = lambda x: x.toarray() if hasattr(x, "toarray") else x


@jax.jit
@partial(jax.vmap, in_axes=[None, 0])
def masked_max(x, mask):
    return jnp.max(x * mask, axis=1)


@jax.jit
@partial(jax.vmap, in_axes=[None, 0])
def masked_mean(x, mask):
    return jnp.sum(x * mask, axis=1) / jnp.sum(mask)


@jax.jit
def frac_nonzero(x, axis=0):
    return jnp.mean(x > 0, axis=axis)


def dict_to_binary(d):
    df = pd.concat(
        [pd.Series(v, name=k).astype(str) for k, v in d.items()],
        axis=1,
    )
    marker_mat = pd.get_dummies(df.stack()).groupby(level=1).sum().clip(upper=1)
    return marker_mat


@numba.jit
def match(a, b):
    return [b.index(x) if x in b else None for x in a]


def get_expr(adata, features=None, layer=None):
    """Get expression matrix from adata object"""
    if features is not None:
        # intersect with adata features
        features = list(set(features) & set(adata.var_names))
        adata = adata[:, match(features, adata.var_names.tolist())]
    else:
        features = adata.var_names.tolist()

    if layer is not None:
        expr = jnp.array(to_dense(adata.layers[layer]))
    else:
        expr = jnp.array(to_dense(adata.X))

    return expr, features


def get_markers(x):
    return {n: v["marker_genes"] for n, v in x.items()}


def read_yaml(file):
    with open(file, "r") as f:
        marker_dict = yaml.safe_load(f)
    return marker_dict


def get_annot_df(x, group_name, min_expr=0.2):
    # Get valid annots from each level
    annot_list = []
    for k, v in x.items():
        annot = v.set_index(group_name)["class"]
        if min_expr > 0:
            expr = v.set_index(group_name)["expr"]
            annot = annot[expr > min_expr]
        annot_list.append(annot)
    # Concat annots
    annot_df = pd.concat(annot_list, axis=1)
    # Rename cols to levels
    annot_df.columns = [str(i) for i in x.keys()]
    return annot_df
