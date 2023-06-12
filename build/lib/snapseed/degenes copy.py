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
    level=None,
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

    result_df_zscore = pd.DataFrame(marker_dict.keys())
    result_df_zscore = result_df_zscore.rename(columns={0:'level_name'})

    result_df_apvalue = pd.DataFrame(marker_dict.keys())
    result_df_apvalue = result_df_apvalue.rename(columns={0:'level_name'})

    if len(adata.obs[group_name].unique()) == 1:
        assign_df = pd.DataFrame(index=adata.obs[group_name].unique())
        assign_df['max_de'] = 'na'
        assign_df['de_score'] = 0
        z_df = pd.DataFrame(columns=adata.obs[group_name].unique(), 
                            index=list(marker_dict.keys())).fillna(0)

    else:
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
        assign_df = assign_df.rename(columns={0:'max_de'})
        assign_df['de_score'] = z_df.max()

    # cal max exp
    raw_bulk = get_bulk_exp(adata, group_name, 'raw')
    max_exp=pd.DataFrame(index=raw_bulk.columns)
    for cell in marker_dict:
        if sum(raw_bulk.index.isin(marker_dict[cell]))>0:
            good_markers = [i for i in marker_dict[cell] if i in raw_bulk.index]
            max_exp[cell] = raw_bulk.loc[good_markers].max()
        else:
            max_exp[cell] = 0

    s = max_exp.select_dtypes(include='object').columns
    max_exp[s] = max_exp[s].astype("float")
    max_exp_df = pd.DataFrame(max_exp.max(axis=1))
    max_exp_df = max_exp_df.rename(columns={0:'exp_score'})

    max_exp_df['max_exp'] = max_exp.idxmax(axis=1)


    # merge de and exp
    assign_df = pd.merge(assign_df, max_exp_df, left_index=True, right_index=True)



    mt_results = z_df * max_exp.T.loc[z_df.index, z_df.columns]

    mt_results_df = pd.DataFrame(mt_results.max(axis=0))
    mt_results_df = mt_results_df.rename(columns={0:'mt_score'})

    mt_results_df['max_mt'] = mt_results.idxmax(axis=0)
    
    assign_df = pd.merge(assign_df, mt_results_df, left_index=True, right_index=True)

    use_mt = True
    na_cutoff = 0.1
    if use_mt:
        classs=[]
        for index,row in assign_df.iterrows():
            if row['mt_score'] <= na_cutoff:
                classs.append(row['max_exp'])
            else:
                classs.append(row['max_mt'])

        # else:
        #     for index,row in assign_df.iterrows():
        #         if row['mt_score'] < na_cutoff:
        #             if row['de_score'] > na_cutoff:
        #                 classs.append(row['max_de'])
        #             elif row['exp_score'] > na_cutoff:
        #                 classs.append(row['max_exp'])
        #             else:
        #                 classs.append('na')
        #         else:
        #             classs.append(row['max_mt'])

    else:
        classs=[]
        # for index,row in assign_df.iterrows():
        #     if row['de_score'] > 2:
        #         classs.append(row['max_de'])
        #     elif row['de_score'] > 1 and row['exp_score'] < 2:
        #         classs.append(row['max_de'])
        #     elif row['de_score'] > 1 and row['exp_score'] > 2:
        #         classs.append(row['max_exp'])
        #     elif row['exp_score'] > 0:
        #         classs.append(row['max_exp'])
        #     else:
        #         classs.append('na')
        for index,row in assign_df.iterrows():
            if row['de_score'] > 2:
                classs.append(row['max_de'])
            elif row['de_score'] > 1:
                if row['exp_score'] > 2:
                    classs.append(row['max_exp'])
                else:
                    classs.append(row['max_de'])
            elif row['de_score'] > 0:
                if row['exp_score'] > 0.5:
                    classs.append(row['max_exp'])
                else:
                    classs.append('na')
            else:
                classs.append('na')

    assign_df['class'] = classs

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

