import math
from functools import partial

import numpy as np
import pandas as pd
from sklearn import preprocessing

import jax
from jax import numpy as jnp

import scanpy as sc

from .auroc import annotate_snap

# from snapseed.utils import read_yaml
from .utils import read_yaml, dict_to_binary, frac_nonzero, masked_max, match, to_dense


def annotate_degenes(
    adata,
    marker_dict,
    group_name,
    layer=None,
    level=None,
    auc_weight=0.5,
    expr_weight=0.5
    ):
    """
    Annotate cell types based on differentially expressed (DE) marker genes.

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
    # level_name = "level_" + str(level)

    # TODO magic way for adata only has one cluster
    # fixed
    # if len(adata.obs[group_name].unique()) <= 1:
    #     # 1st way
    #     # assign_df = annotate_snap(
    #     #     adata, marker_dict, group_name, layer=layer
    #     # )
    #     # 2nd way
    #     assign_df = pd.DataFrame({'class':['na'], 'score':[np.nan], 'expr':[1]})
    #     assign_df.index=adata.obs[group_name].unique()
    #     # 3rd way
    #     # assign_df=pd.DataFrame()
    #     return assign_df
    

    # for celltype in marker_dict['subtypes']:
    #     subgenes = marker_dict['subtypes'][celltype]
    #     if marker_dict['subtypes'][celltype]['marker_genes'] == []:
    #         markers_all=[]
    #         for i in marker_dict['subtypes']:
    #             for j in marker_dict['subtypes'][i]['marker_genes']:
    #                 markers_all.append(j)
    #         marker_dict['subtypes'][celltype]['marker_genes'] = markers_all

    # cal max de
    corr_df = get_bulk_exp(adata, group_name).astype(float).corr()
    corr_df = 1 - corr_df

    if level==1:
        dist_pect=10
    else:
        dist_pect=2
        
    ntop = math.ceil(len(adata.obs[group_name].unique())/dist_pect)
    cluster_to_compair = corr_df.apply(lambda s: s.abs().nlargest(ntop).index.tolist(), axis=1).to_dict()
    for i in cluster_to_compair:
        if i in cluster_to_compair[i]:
            cluster_to_compair[i].remove(i)

    # Reformat marker_dict into binary matrix
    marker_mat = dict_to_binary(marker_dict)

    # Compute AUROC and fraction nonzero for marker features
    features = marker_mat.columns

    le = preprocessing.LabelEncoder()
    le.fit(adata.obs[group_name])
    cell_groups = jnp.array(le.transform(adata.obs[group_name]))

    groups = le.classes_
    
    expr, features = get_expr(adata, features=features, layer=layer)

    aurocs = jnp.zeros((len(groups), len(features)))
    frac_nzs = jnp.zeros((len(groups), len(features)))
    allgroups=[]

    aurocs = pd.DataFrame(aurocs).T
    aurocs.columns=groups
    aurocs.index = features

    frac_nzs = pd.DataFrame(frac_nzs).T
    frac_nzs.columns=groups
    frac_nzs.index = features

    for group in groups:
        
        group = str(group)
        
        adata0 = adata.copy()
        adata0.obs[group_name] = adata0.obs[group_name].astype(str)

        if group not in cluster_to_compair:
            adata0 = adata0[adata0.obs[group_name].isin([group])]

        else:
            adata0.obs.loc[adata0.obs[group_name].isin(cluster_to_compair[group]), 
                        group_name] = 'ref'
            adata0 = adata0[adata0.obs[group_name].isin(['ref', group])]

        metric = auc_expr(adata0, group_name, features=features)
        aurocs[group] = metric['auroc'][metric['groups']==group][0]
        frac_nzs[group] = metric['frac_nonzero'][metric['groups']==group][0]
        allgroups.append(group)

    aurocs=aurocs.fillna(0.5) ### MAGIC TODO MAY PROBLEM
    ### The PDAC Hwang_NatGenet_2022 GSE202051_003 sample
    ### Level 2 mesenchyme only have 1 cluster lead error
    ### KeyError: "None of [Float64Index([nan], dtype='float64')] are in the [columns]"

    metrics={'frac_nonzero':frac_nzs,
            'auroc':aurocs,
            'features':features,
            'groups':groups}


    marker_mat = marker_mat.loc[:, metrics["features"]]

    auc_max = pd.DataFrame()
    for i in marker_dict:
        auc_max[i] = aurocs.loc[[i for i in marker_dict[i] if i in aurocs.index],:].max()

    expr_max = pd.DataFrame()
    for i in marker_dict:
        expr_max[i] = frac_nzs.loc[[i for i in marker_dict[i] if i in frac_nzs.index],:].max()

    # Combine metrics
    assignment_scores = (auc_weight * auc_max + expr_weight * expr_max) / (
        auc_weight + expr_weight
    )

    assign_class = assignment_scores.idxmax(1)

    assign_df = pd.DataFrame(
        {
            "class": assign_class,
            "score": np.diag(assignment_scores[assign_class]),
            "auc": np.diag(auc_max[assign_class]),
            "expr": np.diag(expr_max[assign_class]),
        },
        index=metrics["groups"],
    )

    return assign_df

    # adata.obs[level_name] = adata.obs[group_name].map(cluster2ct)
    # return adata

def get_bulk_exp(adata, bulk_labels, layer='var'):
    if layer=='raw':
        res = pd.DataFrame(columns=adata.raw.var_names, index=adata.obs[bulk_labels].cat.categories)
    else:
        res = pd.DataFrame(columns=adata.var_names, index=adata.obs[bulk_labels].cat.categories)                                                                                                 
    
    for clust in adata.obs[bulk_labels].cat.categories: 
        if layer=='raw':
            res.loc[clust] = adata[adata.obs[bulk_labels].isin([clust]),:].raw.X.mean(0)
        else:
            res.loc[clust] = adata[adata.obs[bulk_labels].isin([clust]),:].X.mean(0)

    res.index=adata.obs[bulk_labels].cat.categories

    return res.T

def get_expr(adata, features=None, layer=None):
    """Get expression matrix from adata object"""

    if layer == 'raw' or layer == None:
        adata = adata.raw.to_adata()

    if features is not None:
        # intersect with adata features
        features = list(set(features) & set(adata.var_names))
        adata = adata[:, match(features, adata.var_names.tolist())]

    if layer == 'raw' or layer == None:
        expr = jnp.array(to_dense(adata.X))

    else:
        expr = jnp.array(to_dense(adata.layers[layer]))

    return expr, features

def auc_expr(adata, group_name, features=None, layer=None):
    """Computes AUROC and fraction nonzero for each gene in an adata object."""
    # Turn string groups into integers
    le = preprocessing.LabelEncoder()
    le.fit(adata.obs[group_name])
    # Compute AUROC and fraction nonzero
    cell_groups = jnp.array(le.transform(adata.obs[group_name]))
    
    groups = le.classes_

    expr, features = get_expr(adata, features=features, layer=layer)
    auroc, frac_nonzero = expr_auroc_over_groups(expr, cell_groups, groups)

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


def expr_auroc_over_groups(expr, cell_groups, groups):
    """Computes AUROC for each group separately."""
    auroc = jnp.zeros((len(groups), expr.shape[1]))
    frac_nz = jnp.zeros((len(groups), expr.shape[1]))

    for group in groups:
        if group == 'ref':
            group = 0
        else: 
            group = int(group)
        auroc = auroc.at[group, :].set(jit_auroc(expr, cell_groups == group))
        frac_nz = frac_nz.at[group, :].set(frac_nonzero(expr[cell_groups == group, :]))

    return auroc, frac_nz

