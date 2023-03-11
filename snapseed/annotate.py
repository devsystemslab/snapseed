import yaml
import pandas as pd

from .trinarize import annotate_cytograph
from .auroc import annotate_snap

from .utils import get_subtypes


def annotate_hierarchy(adata, marker_yaml, group_name, method="auroc", layer=None):
    """Annotate clusters hierarchically with marker genes."""
    # Load marker yaml
    with open(marker_yaml, "r") as f:
        marker_hierarchy = yaml.safe_load(f)
    # Annotate at each level of the hierarchy
    assignment_hierarchy = annotate_subtypes(
        adata, marker_hierarchy, group_name, method=method
    )


def annotate_subtypes(adata, marker_hierarchy, group_name, method="auroc", layer=None):
    """Recursively annotatates all"""
    subtype_markers, subtype_hierarchy = get_subtypes(marker_hierarchy)
    for subtype in subtype_markers.keys():
        # Get marker genes for this level
        marker_dict = subtype_markers[subtype]
        # Annotate clusters
        assignments = annotate(
            adata, marker_dict, group_name, method=method, layer=layer
        )
        # Iterate through annotated types and subset adata
        subtype_assignments = []
        for annot_subtype in assignments["class"].unique():
            if annot_subtype not in subtype_hierarchy.keys():
                continue
            # Subset adata
            subtype_groups = assignments.index[assignments["class"] == subtype]
            subtype_adata = adata[adata.obs[group_name] == subtype_groups]
            # Recursively annotate
            subtype_annots = annotate_subtypes(
                subtype_adata,
                subtype_hierarchy[subtype],
                group_name,
                method=method,
                layer=layer,
            )
            subtype_assignments.append(subtype_annots)
        # Join subtype assignments
        subtype_assignments = pd.concat(subtype_assignments, axis=0)
        return [assignments] + subtype_assignments


def annotate(adata, marker_dict, group_name, method="auroc", layer=None):
    """Annotate clusters with marker genes."""
    if method == "auroc":
        assignments = annotate_snap(adata, marker_dict, group_name, layer=layer)
    if method == "trinatize":
        assignments = annotate_cytograph(adata, marker_dict, group_name, layer=layer)
    else:
        raise ValueError("Unknown method.")
    # Join cluster-level results with adata
    assignments = assignments.reset_index(names=group_name)
    return assignments
