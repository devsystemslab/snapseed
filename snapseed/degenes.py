import math

import numpy as np
import pandas as pd

import scanpy as sc
from .auroc import annotate_snap

from snapseed.utils import read_yaml


def annotate_degenes(
    adata,
    marker_dict,
    group_name,
    layer=None,
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
    print(adata.obs[group_name].unique())
    if len(adata.obs[group_name].unique()) <= 1:
        # 1st way
        # assign_df = annotate_snap(
        #     adata, marker_dict, group_name, layer=layer
        # )
        # 2nd way
        assign_df = pd.DataFrame({'class':['na'], 'score':[np.nan], 'expr':[1]})
        assign_df.index=adata.obs[group_name].unique()
        # 3rd way
        # assign_df=pd.DataFrame()
        return assign_df
    
    corr_df = get_bulk_exp(adata, group_name).astype(float).corr()
    corr_df = 1 - corr_df

    ntop = math.ceil(len(adata.obs[group_name].unique())/10)
    cluster_to_compair = corr_df.apply(lambda s: s.abs().nlargest(ntop).index.tolist(), axis=1).to_dict()

    result_df_zscore = pd.DataFrame(marker_dict.keys())
    result_df_zscore = result_df_zscore.rename(columns={0:'level_name'})

    result_df_apvalue = pd.DataFrame(marker_dict.keys())
    result_df_apvalue = result_df_apvalue.rename(columns={0:'level_name'})

    for cluster in adata.obs[group_name].unique():
        adata0 = adata.copy()
        adata0.obs[group_name] = adata0.obs[group_name].astype(str)

        # sc.tl.rank_genes_groups(adata0, group_name, groups=[cluster], 
        #                         reference=cluster_to_compair[cluster], method='wilcoxon')
        
        adata0.obs.loc[adata0.obs[group_name].isin(cluster_to_compair[cluster]), 
                       group_name] = 'ref'

        sc.tl.rank_genes_groups(adata0, group_name, groups=[cluster], 
                                reference='ref', method='wilcoxon')

        wranks = wrangle_ranks_from_adata(adata0)

        z_scores=[]
        adj_pvalss=[]
        for i in marker_dict:
            z_scores.append(wranks.loc[wranks.gene.isin(marker_dict[i]), 'z_score'].max())
            adj_pvalss.append(-np.log10(wranks.loc[wranks.gene.isin(marker_dict[i]), 'adj_pvals']).max())
            # z_scores.append(wranks.loc[wranks.gene.isin(marker_dict[i]['marker_genes']), 'z_score'].max())
            # adj_pvalss.append(-np.log10(wranks.loc[wranks.gene.isin(marker_dict[i]['marker_genes']), 'adj_pvals']).max())

        # result_df_zscore[cluster]=[i*j for i,j in zip(z_scores,adj_pvalss)]
        result_df_zscore[cluster]=z_scores
        result_df_apvalue[cluster]=adj_pvalss

    z_df = result_df_zscore.set_index('level_name')
    cluster2ct = z_df.idxmax().to_dict()

    has_na = False
    # TODO add has_na
    if has_na:    
        for i in cluster2ct:
            if i in z_df[z_df.max(axis=1)<1].index:
                cluster2ct[i] = 'na'
                
    assign_df = pd.DataFrame(pd.Series(cluster2ct))
    assign_df = assign_df.rename(columns={0:'class'})
    assign_df['score'] = z_df.max()

    # TODO magic way to avoid get_annot_df error
    assign_df['expr'] = 1

    return assign_df

    # adata.obs[level_name] = adata.obs[group_name].map(cluster2ct)
    # return adata

def wrangle_ranks_from_adata(adata):
    """
    Wrangle results from the ranked_genes_groups function of Scanpy.
    """
    # Get number of top ranked genes per groups
    nb_marker = len(adata.uns['rank_genes_groups']['names'])
    # Wrangle results into a table (pandas dataframe)
    top_score = pd.DataFrame(adata.uns['rank_genes_groups']['scores']).loc[:nb_marker]
    top_adjpval = pd.DataFrame(adata.uns['rank_genes_groups']['pvals_adj']).loc[:nb_marker]
    top_gene = pd.DataFrame(adata.uns['rank_genes_groups']['names']).loc[:nb_marker]
    marker_df = pd.DataFrame()
    # Order values
    for i in top_score.columns:
        concat = pd.concat([top_score[[str(i)]], top_adjpval[str(i)], top_gene[[str(i)]]], axis=1, ignore_index=True)
        concat['cluster_number'] = i
        col = list(concat.columns)
        col[0], col[1], col[-2] = 'z_score', 'adj_pvals', 'gene'
        concat.columns = col
        marker_df = marker_df.append(concat)
    return marker_df

def get_bulk_exp(adata, bulk_labels):
    res = pd.DataFrame(columns=adata.var_names, index=adata.obs[bulk_labels].cat.categories)                                                                                                 

    for clust in adata.obs[bulk_labels].cat.categories: 
        res.loc[clust] = adata[adata.obs[bulk_labels].isin([clust]),:].X.mean(0)

    res.index=adata.obs[bulk_labels].cat.categories

    return res.T

