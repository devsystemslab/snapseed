import jax
from jax import numpy as jnp


def annotate(adata, marker_list, group_name, method="snap"):
    """Annotate clusters with marker genes."""
    if method == "snap":
        return annotate_snap(adata, marker_list, group_name)


def annotate_snap(adata, marker_dict, group_name):
    """
    Anntoate cell types based on AUROC and expression of predefined marker genes.

    Parameters
    ----------
    adata
        AnnData object
    marker_dict
        Dict with marker genes for each celltype
    group_name
        Name of the column in adata.obs that contains the cluster labels
    """
    # Compute AUROC and fraction nonzero for marker features
    features = list(set.union(*[set(x) for x in marker_dict.values()]))
    metrics = auc_expr(adata, group_name, features=features)

    # Reformat marker_dict into binary matrix
    df = pd.concat(
        [pd.Series(v, name=k).astype(str) for k, v in marker_dict.items()],
        axis=1,
    )
    marker_mat = pd.get_dummies(df.stack()).sum(level=1).clip_upper(1)


#

### Old R code
# assign_class <- function(object, marker_list, cluster_name){

#     marker_mat <- marker_list %>%
#         enframe('class', 'gene') %>%
#         unnest_longer(gene) %>%
#         mutate(value=1) %>%
#         pivot_wider(names_from='class', values_fill=0) %>%
#         column_to_rownames('gene') %>% as.matrix()

#     marker_auc <- cluster_de %>%
#         filter(feature%in%rownames(marker_mat)) %>%
#         dplyr::select(feature, group, auc) %>%
#         pivot_wider(names_from=group, values_from=auc) %>%
#         column_to_rownames('feature') %>% as.matrix()

#     marker_ex <- cluster_de %>%
#         filter(feature%in%rownames(marker_mat)) %>%
#         dplyr::select(feature, group, prcex_self) %>%
#         pivot_wider(names_from=group, values_from=prcex_self) %>%
#         column_to_rownames('feature') %>% as.matrix() %>% {./100}

#     max_auc_mat <- map(names(marker_list), function(n){
#         x <- marker_list[[n]]
#         auc_mat <- marker_auc[intersect(x, rownames(marker_auc)), ]
#         if (is.null(ncol(auc_mat))){
#             return(enframe(auc_mat, 'group', n))
#         } else {
#             max_mat <- colMaxs(auc_mat)
#             names(max_mat) <- colnames(marker_auc)
#             return(enframe(max_mat, 'group', n))
#         }
#     }) %>% reduce(inner_join) %>% column_to_rownames('group') %>% as.matrix()

#     max_auc_df <- max_auc_mat %>%
#         as_tibble(rownames='group') %>%
#         pivot_longer(!group, names_to='class', values_to='auc')

#     max_ex_mat <- map(names(marker_list), function(n){
#         x <- marker_list[[n]]
#         ex_mat <- marker_ex[intersect(x, rownames(marker_ex)), ]
#         if (is.null(ncol(ex_mat))){
#             return(enframe(ex_mat, 'group', n))
#         } else {
#             max_mat <- colMaxs(ex_mat)
#             names(max_mat) <- colnames(marker_ex)
#             return(enframe(max_mat, 'group', n))
#         }
#     }) %>% reduce(inner_join) %>% column_to_rownames('group') %>% as.matrix()

#     max_ex_df <- max_ex_mat %>%
#         as_tibble(rownames='group') %>%
#         pivot_longer(!group, names_to='class', values_to='expr_frac')

#     score_mat <- max_auc_mat * max_ex_mat
#     class_assign <- colnames(score_mat)[apply(score_mat, 1, which.max)]
#     class_score <- rowMaxs(score_mat)
#     class_assign_df <- tibble(
#         pred_class = class_assign,
#         pred_score = class_score,
#         group = rownames(score_mat)
#     )

#     return(list(assignment=class_assign_df, auc=max_auc_mat, expr=max_ex_mat))
# }
