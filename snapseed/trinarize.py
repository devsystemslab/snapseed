import numba
import jax
from jax import numpy as jnp
import pandas as pd
import numpy as np


from jax.scipy.special import gammaln, betainc, betaln

from .utils import dict_to_binary


def annotate_cytograph(adata, marker_dict, group_name, layer=None, f=0.2):
    """Annotate clusters based on trinarization of marker gene expression."""
    # Reformat marker_dict into binary matrix
    marker_mat = dict_to_binary(marker_dict)
    # Compute AUROC and fraction nonzero for marker features
    features = marker_mat.columns
    expr, features = get_expr(adata, features=features, layer=layer)
    marker_mat = marker_mat.loc[:, features]
    marker_array = marker_mat.values
    #
    le = preprocessing.LabelEncoder()
    le.fit(adata.obs[group_name])
    groups = jnp.array(le.transform(adata.obs[group_name]))
    n_groups = int(groups.max() + 1)
    # trinatize
    trinaries = betabinomial_trinarize_array(expr, groups, n_groups, f)
    annot_probs = get_annot_probs(np.array(trinaries), marker_array)
    annot_dict = {}
    annot_scores = np.zeros(annot_probs.shape[1])
    for i in range(annot_probs.shape[1]):
        annot_dict[le.classes_[i]] = marker_mat.index[np.argmax(annot_probs[:, i])]
        annot_scores[i] = np.max(annot_probs[:, i])
    annot_tags = [
        tag if isinstance(tag, str) else ";".join(tag) for tag in annot_dict.values()
    ]
    assign_df = pd.DataFrame(
        {"class": annot_tags, "score": annot_scores},
        index=annot_dict.keys(),
    )
    return assign_df


@numba.jit
def get_annot_probs(trinaries, marker_array):
    """Compute the annotaion probability foea each cell type."""
    group_probs = np.zeros((marker_array.shape[0], trinaries.shape[1]))
    for i in np.arange(trinaries.shape[1]):
        for j in np.arange(marker_array.shape[0]):
            marker_trinaries = trinaries[:, i][np.nonzero(marker_array[j, :])]
            if marker_trinaries.shape[0] == 0:
                group_probs[j, i] = 0
            else:
                group_probs[j, i] = np.mean(marker_trinaries)
    return group_probs


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

annotate_cytograph(adata, marker_dict, group_name, layer="counts", f=0.2)
