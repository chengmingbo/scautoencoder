# single cell autoencoder


#example for scanpy anndata

```python
import scanpy as sc
import scautoencoder as scae
import pandas as pd
import numpy as np


adata = sc.datasets.pbmc3k()

adata.var_names_make_unique()

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
)

adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :].copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

adata = adata[:, adata.var.highly_variable]

scae.tl.autoencoder(adata)

sc.pp.neighbors(adata, use_rep='X_encoded')
sc.tl.umap(adata)
```
