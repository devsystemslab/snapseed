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


def get_markers(x):
    return {n: v["marker_genes"] for n, v in x.items()}


def get_subtypes(x):
    subtype_dict = {}
    marker_dicts = {}
    for k, v in x.items():
        if "subtypes" in v.keys():
            subtype_dict[k] = v["subtypes"]
            marker_dicts[k] = yaml_to_dict(v["subtypes"])
    return subtype_dict, marker_dicts
