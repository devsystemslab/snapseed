import pandas as pd

from .trinarize import annotate_cytograph
from .auroc import annotate_snap

from .utils import get_subtypes, get_markers


def annotate_hierarchy(adata, marker_hierarchy, group_name, method="auroc", layer=None):
    """Annotate clusters hierarchically with marker genes."""

    # Annotate at each level of the hierarchy
    assignment_hierarchy = annotate_subtypes(
        adata, marker_hierarchy, group_name, method=method
    )

    return assignment_hierarchy


def annotate_subtypes(adata, marker_hierarchy, group_name, method="auroc", layer=None):
    """Recursively annotatates all"""
    marker_dict = get_markers(marker_hierarchy)
    assignments = annotate(adata, marker_dict, group_name, method=method, layer=layer)
    assignment_list = [assignments]
    for subtype in assignments["class"].unique():

        if "subtypes" not in marker_hierarchy[subtype].keys():
            continue

        # Subset adata
        subtype_groups = assignments[group_name][
            assignments["class"] == subtype
        ].astype(str)
        subtype_adata = adata[adata.obs[group_name].isin(subtype_groups)]

        # Recursively annotate
        subtype_assignments = annotate_subtypes(
            subtype_adata,
            marker_hierarchy[subtype]["subtypes"],
            group_name,
            method=method,
            layer=layer,
        )
        assignment_list += subtype_annots

    return assignment_list


def annotate(adata, marker_dict, group_name, method="auroc", layer=None):
    """Annotate clusters with marker genes."""
    if method == "auroc":
        assignments = annotate_snap(adata, marker_dict, group_name, layer=layer)
    elif method == "trinatize":
        assignments = annotate_cytograph(adata, marker_dict, group_name, layer=layer)
    else:
        raise ValueError("Unknown annotation method.")
    # Join cluster-level results with adata
    assignments = assignments.reset_index(names=group_name)
    return assignments
