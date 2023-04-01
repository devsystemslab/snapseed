import math

import numpy as np
import pandas as pd

import scanpy as sc

from snapseed.utils import read_yaml


def annotate_cluster(
    adata,
    marker_dict,
    group_name,
    layer=None,
    level_name=None
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
    if level_name=='level_1':
        adata=annot_adata(adata, marker_genes, level_name='level_1', has_na=False)
    else:
        adata=annot_adata_lowlevel(adata, marker_genes, level_name=level_name, has_na=True)


def annot_adata(
    adata, 
    marker_genes, 
    level_name='level_1', 
    has_na=False, 
    group_name=group_name
    ):

    corr_df = get_bulk_exp(adata, group_name).astype(float).corr()
    corr_df = 1 - corr_df

    ntop = math.ceil(len(adata.obs[group_name].unique())/10)
    cluster_to_compair = corr_df.apply(lambda s: s.abs().nlargest(ntop).index.tolist(), axis=1).to_dict()

    result_df_zscore = pd.DataFrame(marker_genes.keys())
    result_df_zscore = result_df_zscore.rename(columns={0:level_name})

    result_df_apvalue = pd.DataFrame(marker_genes.keys())
    result_df_apvalue = result_df_apvalue.rename(columns={0:level_name})

    for cluster in adata.obs[group_name].unique():
        adata0 = adata.copy()
        adata0.obs[group_name] = adata0.obs[group_name].astype(str)

        # sc.tl.rank_genes_groups(adata0, group_name, groups=[cluster], 
        #                         reference=cluster_to_compair[cluster], method='wilcoxon')
        
        adata0.obs.loc[adata0.obs[group_name].isin(cluster_to_compair[cluster]), 
                       group_name] = 'ref'

        sc.tl.rank_genes_groups(adata0, f"leiden_{high_res}", groups=[cluster], 
                                reference='ref', method='wilcoxon')

        wranks = wrangle_ranks_from_adata(adata0)

        z_scores=[]
        adj_pvalss=[]
        for i in marker_genes:
            z_scores.append(wranks.loc[wranks.gene.isin(marker_genes[i]['marker_genes']), 'z_score'].max())
            adj_pvalss.append(-np.log10(wranks.loc[wranks.gene.isin(marker_genes[i]['marker_genes']), 'adj_pvals']).max())

        # result_df_zscore[cluster]=[i*j for i,j in zip(z_scores,adj_pvalss)]
        result_df_zscore[cluster]=z_scores
        result_df_apvalue[cluster]=adj_pvalss

    z_df = result_df_zscore.set_index(level_name)
    cluster2ct = z_df.idxmax().to_dict()

    if has_na:    
        for i in cluster2ct:
            if i in z_df[z_df.max(axis=1)<1].index:
                cluster2ct[i] = 'na'

    adata.obs[level_name] = adata.obs[group_name].map(cluster2ct)

    return adata

def annot_adata_level2(
    adata, 
    marker_genes, 
    level_name='level_2', 
    has_na=True
    ):

    level2={}
    for i in adata.obs.level_1.unique():
        sub_adata = adata[adata.obs.level_1==i]
        
        if 'subtypes' not in marker_genes[i]:
            level2.update(sub_adata.obs.level_1.to_dict())
        else:
            sub_adata = sub_adata.copy()
            sub_adata = annot_adata(sub_adata, marker_genes[i]['subtypes'],
                                  level_name='level_2', has_na=True)
            sub_adata = sub_adata[sub_adata.obs.level_1==i]

            level2.update(sub_adata.obs.level_2.to_dict())
            
    adata.obs['level_2'] = adata.obs.index.map(level2)
    
    return adata

def annot_adata_lowlevel(
    adata, 
    marker_genes, 
    level_name='level_2', 
    has_na=True
    ):

    llevel = {}
    up_level_name = f"level_{int(level_name.split('_')[-1])-1}"

    for i in adata.obs[up_level_name].unique():
        sub_adata = adata[adata.obs[up_level_name] == i].copy()
        
        if 'subtypes' not in marker_genes[i]:
            llevel.update(sub_adata.obs[up_level_name].to_dict())
        else:
            sub_adata = annot_adata(sub_adata, marker_genes[i]['subtypes'],
                                  level_name=level_name, has_na=True)
            sub_adata = sub_adata[sub_adata.obs[up_level_name] == i]

            llevel.update(sub_adata.obs[level_name].to_dict())
            
    adata.obs[level_name] = adata.obs.index.map(llevel)
    
    return adata

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

