#  **CAR T cell foundation model predicts immunotherapy response**

### **Overview**
Single-cell transcriptomics provides unprecedented resolution for characterizing CAR T-cell states, yet translating heterogeneous cellular signals into accurate patient-level therapeutic response remains a major challenge. Existing studies primarily rely on experimental and statistical analyses to identify differentially expressed genes or cell populations associated with response, but lack predictive frameworks that systematically integrate gene-level structure with patient-level outcomes. Here, we present gANCHOR, a hierarchical hypergraph attention framework that integrates biologically informed representation learning with patient-level response prediction. By encoding gene-pathway relationships, gANCHOR learns pathway-aware cell embeddings that improve biological conservation and batch robustness across datasets. A cell-to-patient attention module aggregates cellular information to infer therapeutic response. Under a unified downstream framework, gANCHOR achieves superior prediction performance compared to existing foundation models. The model also identifies reproducible gene signatures associated with response and non-response, providing interpretable insights into relevant cellular programs.

<p align="center">  <img src="https://https://github.com/luoyuanlab/cart/tree/main/img_folder" height="800px" />  </p>  <br />

### **Benchmark on real datasets**
To systematically evaluate current models, we benchmarked them on two well-characterized datasets, mouse gastrulation erythroid lineage and mouse dentate gyrus, both of which are widely used due to their rich biological annotation and challenging dynamic structures. The dentate gyrus neurogenesis single cell data can be extracted using [scVelo](https://scvelo.readthedocs.io/en/stable/index.html) command [scvelo.datasets.dentategyrus()](https://scvelo.readthedocs.io/en/stable/scvelo.datasets.dentategyrus.html). The mouse gastrulation erythroid lineage data can be extracted via [scVelo](https://scvelo.readthedocs.io/en/stable/index.html) command [scvelo.datasets.gastrulation_erythroid()](https://scvelo.readthedocs.io/en/stable/scvelo.datasets.gastrulation_erythroid.html).

### **Benchmark on simulated dataset: tumor microenvironment**
To further assess model performance under spatially heterogeneous kinetics, we simulated 4,000 CD8⁺ T cells (2,000 from immune-sensitive and 2,000 from immune-resistant regions) focusing on _PTPN11_, _KRAS_, and _MAPK1_. Among the three genes, only _PTPN11_ in the immune-sensitive area exhibits gradually changing transcription, whereas _KRAS_ and _MAPK1_ in the immune-sensitive area, and all three genes in the immune-resistant area display abrupt shifts in their transcription rates (_α_) at specific time points such as T-cell activation or PD-1/PD-L1 binding (Figure 2). These sharp kinetic transitions are particularly challenging for conventional RNA velocity models, which typically assume smooth or even time-invariant transcription rates, to capture. The simulated dataset in stored in the `Data/` directory, while the scripts in `simulation/` can be used to regenerate them.

<p align="center">  <img src="https://github.com/luoyuanlab/rna_velocity/blob/main/img_folder/Figure%203.png" height="800px" />  </p>  <br />

## **System requirements**
### **OS Requirements**
The codes have been tested on the following systems:
* Linux: CentOS Linux 7
* Linux: Red Hat Enterprise Linux 8.10

### **Python Dependencies**
Expected installation time: ~15 minutes.
<pre>
numpy
scipy
torch
pandas
scanpy
anndata
pickle
seaborn
umap
matplotlib
</pre>

