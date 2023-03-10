from .trinarize import annotate_cytograph
from .auroc import annotate_snap


def annotate(adata, marker_dict, group_name, method="snap", layer=None):
    """Annotate clusters with marker genes."""
    if method == "auroc":
        assignments = annotate_snap(adata, marker_dict, group_name, layer=layer)
    if method == "trinatize":
        assignments = annotate_cytograph(adata, marker_dict, group_name)
    else:
        raise ValueError("Unknown method.")
    # Join cluster-level results with adata
    assignments = assignments.reset_index(names=group_name)
    return assignments
