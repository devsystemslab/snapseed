import pandas as pd

import jax
from jax import numpy as jnp

from jax.scipy.special import gammaln, betainc, betaln

from .utils import auc_expr, dict_to_binary


def trinarize(expr, group_name, marker_dict, f=0.2, layer=None):
    # Turn string groups into integers
    le = preprocessing.LabelEncoder()
    le.fit(adata.obs[group_name])
    groups = jnp.array(le.transform(adata.obs[group_name]))
    n_groups = int(groups.max() + 1)
    # Subset features from marker_dict items
    features = set([k for v in marker_dict.values() for k in v])
    features = list(features & set(adata.var_names))
    # get expression matrix
    if layer is not None:
        expr = jnp.array(to_dense(adata[:, features].layers[layer]))
    else:
        expr = jnp.array(to_dense(adata[:, features].X))
    # trinatize
    trinary_prob = betabinomial_trinarize_array(expr, groups, n_groups, f)
    return trinary_prob


@partial(jax.jit, static_argnames=["n_groups", "f"])
@partial(jax.vmap, in_axes=[1, None, None, None])
def betabinomial_trinarize_array(x, groups, n_groups, f):
    """
    Trinarize a vector, grouped by groups, using a beta binomial model
    Parameters
    ----------
            x
                The input expression vector.
            groups
                Group labels.

    Returns
    -------
            ps
                The posterior probability of xession in at least a fraction f
    """
    x = jnp.round(jnp.array(x))
    n_by_group = jnp.bincount(groups, length=n_groups)
    k_by_group = jnp.zeros(n_groups)
    for g in jnp.arange(n_groups):
        group_mask = jnp.array(groups == g, dtype=jnp.int32)
        k_by_group = k_by_group.at[g].set(jnp.count_nonzero(x * group_mask))
    ps = p_half(k_by_group, n_by_group, f)
    return ps


@jax.jit
@partial(jax.vmap, in_axes=[0, 0, None])
def p_half(k: int, n: int, f: float) -> float:
    """
    Return probability that at least half the cells express, if we have observed k of n cells expressing:

    p|k,n = 1-(betainc(1+k, 1-k+n, f)*gamma(2+n)/(gamma(1+k)*gamma(1-k+n))/beta(1+k, 1-k+n)

    Parameters
    ----------
    k
        Number of observed positive cells
    n
        Total number of cells
    """
    # These are the prior hyperparameters beta(a,b)
    a = 1.5
    b = 2
    # We really want to calculate this:
    # p = 1-(betainc(a+k, b-k+n, 0.5)*beta(a+k, b-k+n)*gamma(a+b+n)/(gamma(a+k)*gamma(b-k+n)))
    #
    # But it's numerically unstable, so we need to work on log scale (and special-case the incomplete beta)
    incb = betainc(a + k, b - k + n, f)
    p = 1.0 - jnp.exp(
        jnp.log(incb)
        + betaln(a + k, b - k + n)
        + gammaln(a + b + n)
        - gammaln(a + k)
        - gammaln(b - k + n)
    )
    return p


expr = jnp.array(to_dense(adata[:, features].layers["counts"]))
x = expr[:, 0]
group_name = "res.0.6"
k = k_by_group[0]
n = n_by_group[0]


marker_dict = dict(
    Dorsal_Telencephalon=["EMX1", "NEUROD6", "NFIX"],
    Ventral_Telencephalon=["DLX5", "DLX2"],
    Diencephalon=["NHLH2", "LHX5", "RSPO3", "RSPO2"],
    Mesencephalon=["OTX2", "LHX1", "LHX5", "ZIC1"],
    Rhombencephalon=["HOXB2"],
)
