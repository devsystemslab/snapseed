import math

import numpy as np
import pandas as pd

import scanpy as sc

from snapseed.utils import read_yaml


def wrangle_ranks_from_anndata(anndata):
    """
    Wrangle results from the ranked_genes_groups function of Scanpy (Wolf et al., 2018) on louvain clusters.
    """
    # Get number of top ranked genes per groups
    nb_marker = len(anndata.uns['rank_genes_groups']['names'])
    # Wrangle results into a table (pandas dataframe)
    top_score = pd.DataFrame(anndata.uns['rank_genes_groups']['scores']).loc[:nb_marker]
    top_adjpval = pd.DataFrame(anndata.uns['rank_genes_groups']['pvals_adj']).loc[:nb_marker]
    top_gene = pd.DataFrame(anndata.uns['rank_genes_groups']['names']).loc[:nb_marker]
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


def annot_adata(adata, marker_genes, level_name='level_1', 
                has_na=False, res=1.0):

    corr_df = get_bulk_exp(adata, f"leiden_{res}").astype(float).corr()
    corr_df = 1-corr_df
    # cluster_to_compair = corr_df.idxmax().to_dict()
    ntop = math.ceil(len(adata.obs[f"leiden_{res}"].unique())/10)
    cluster_to_compair = corr_df.apply(lambda s: s.abs().nlargest(ntop).index.tolist(), axis=1).to_dict()

    result_df_zscore = pd.DataFrame(marker_genes.keys())
    result_df_zscore = result_df_zscore.rename(columns={0:level_name})

    for cluster in adata.obs[f"leiden_{res}"].unique():
        adata0 = adata.copy()
        adata0.obs[f"leiden_{res}"] = adata0.obs[f"leiden_{res}"].astype(str)

        # sc.tl.rank_genes_groups(adata0, f"leiden_{res}", groups=[cluster], 
        #                         reference=cluster_to_compair[cluster], method='wilcoxon')
        
        adata0.obs.loc[adata0.obs[f"leiden_{res}"].isin(cluster_to_compair[cluster]), 
                       f"leiden_{res}"] = 'ref'

        sc.tl.rank_genes_groups(adata0, f"leiden_{high_res}", groups=[cluster], 
                                reference='ref', method='wilcoxon')

        a= wrangle_ranks_from_anndata(adata0)

        z_scores=[]
        adj_pvalss=[]
        for i in marker_genes:
            z_scores.append(a.loc[a.gene.isin(marker_genes[i]['marker_genes']), 'z_score'].max())
            adj_pvalss.append(-np.log10(a.loc[a.gene.isin(marker_genes[i]['marker_genes']), 'adj_pvals']).max())
        # result_df_zscore[cluster]=[i*j for i,j in zip(z_scores,adj_pvalss)]
        result_df_zscore[cluster]=z_scores

    z_df = result_df_zscore.set_index(level_name)
    cluster2ct = z_df.idxmax().to_dict()
    if has_na:    
        for i in cluster2ct:
            if i in z_df[z_df.max(axis=1)<1].index:
                cluster2ct[i]='na'

    adata.obs[level_name] = adata.obs[f"leiden_{res}"].map(cluster2ct)

    return adata

def annot_adata_level2(adata, marker_genes, level_name='level_2', has_na=True):
    level2={}
    for i in adata.obs.level_1.unique():
        sub_adata = adata[adata.obs.level_1==i]
        
        if 'subtypes' not in marker_genes[i]:
            level2.update(sub_adata.obs.level_1.to_dict())
        else:
            sub_adata = sub_adata.copy()
            sub_adata=annot_adata(sub_adata, marker_genes[i]['subtypes'],
                                  level_name='level_2', has_na=True)
            sub_adata = sub_adata[sub_adata.obs.level_1==i]

            level2.update(sub_adata.obs.level_2.to_dict())
            
    adata.obs['level_2'] = adata.obs.index.map(level2)
    
    return adata

res=1.0
sc.tl.leiden(adata, resolution=res, key_added=f"leiden_{res}")

adata=annot_adata(adata, marker_genes, level_name='level_1', has_na=True)
adata=annot_adata_level2(adata, marker_genes, level_name='level_2', has_na=True)
