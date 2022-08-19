# single-cell data

## mouse data

**current version**: `mouse_220805.h5ad`

- corrected genotype label metadata (`genotype_crct`)
- updated UBC cell type to have labels for 2 cell sub-types (`ctype_ubcupdate`)
- used an updated version of scgat based on insights from You et al. 2021, see [here](https://arxiv.org/pdf/2011.08843.pdf)

### labels 

in `adata.obs`, all metadata per cell is there. Labels of interest include column names:
- *cell type*: `ctype_ubcupdate`
- *genotype*: `genotype_crct`
- *timepoint and genotype*: `genotime_crct`
- *pseudotime*: **#todo**: rerun pseudotime to get continuous, smooth label via graph laplacian