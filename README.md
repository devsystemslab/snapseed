# snapseed <img src="logo/logo.png" align="right" width="220"/>

Snapseed annotates single-cell datasets based on manually defined sets of marker genes for individual cell types or cell type hierarchies. It is fast and simple to accelerate annotation of very large datasets.


## Quick start

```python
import snapseed as snap
from snapseed.utils import read_yaml

# Read in the marker genes
marker_genes = read_yaml("marker_genes.yaml")

# Annotate anndata objects
snap.annotate(
    adata,
    marker_genes,
    group_name="clusters",
    layer="lognorm",
)

# Or for more complex hierarchies
snap.annotate_hierarchy(
    adata,
    marker_genes,
    group_name="clusters",
    layer="lognorm",
)
```
