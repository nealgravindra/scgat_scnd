# single-cell data

## mouse data

**current version**: `mouse_220805.h5ad`

- corrected genotype label metadata (`genotype_crct`)
- updated UBC cell type to have labels for 2 cell sub-types (`ctype_ubcupdate`)

### labels 

in `adata.obs`, all metadata per cell is there. Labels of interest include column names:
- *cell type*: `ctype_ubcupdate`
- *genotype*: `genotype_crct`
- *timepoint*: `timepoint`
- *pseudotime*: **#todo**: rerun pseudotime to get continuous, smooth label via graph laplacian