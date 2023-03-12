import pandas as pd

from .trinarize import annotate_cytograph
from .auroc import annotate_snap

from .utils import get_markers


def annotate_hierarchy(adata, marker_hierarchy, group_name, method="auroc", layer=None):
    """Annotate clusters hierarchically with marker genes."""

    # Annotate at each level of the hierarchy
    assignment_hierarchy = annotate_levels(
        adata, marker_hierarchy, group_name, method=method
    )

    return assignment_hierarchy


def annotate_levels(
    adata,
    marker_hierarchy,
    group_name,
    level=0,
    assignment_levels=None,
    method="auroc",
    layer=None,
):
    """Recursively annotatates all"""
    level += 1
    marker_dict = get_markers(marker_hierarchy)
    assignments = annotate(adata, marker_dict, group_name, method=method, layer=layer)

    if assignment_levels is None:
        assignment_levels = {}

    if level not in assignment_levels.keys():
        assignment_levels[level] = []
    assignment_levels[level].append(assignments)

    for subtype in assignments["class"].unique():

        if "subtypes" not in marker_hierarchy[subtype].keys():
            continue

        # Subset adata
        subtype_groups = assignments[group_name][
            assignments["class"] == subtype
        ].astype(str)
        subtype_adata = adata[adata.obs[group_name].isin(subtype_groups)]

        # Recursively annotate
        assignment_levels = annotate_levels(
            subtype_adata,
            marker_hierarchy[subtype]["subtypes"],
            group_name,
            level=level,
            assignment_levels=assignment_levels,
            method=method,
            layer=layer,
        )

    return assignment_levels


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
