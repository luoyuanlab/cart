<img src="https://github.com/luoyuanlab/cart/blob/main/img_folder/Logo.png" height="160px" />  <br />
#  **CAR T cell foundation model predicts immunotherapy response**

Single-cell transcriptomics provides unprecedented resolution for characterizing CAR T-cell states, yet translating heterogeneous cellular signals into accurate patient-level therapeutic response remains challenging. Existing studies primarily rely on experimental and statistical analyses to identify genes or cell populations associated with response, but lack predictive frameworks that integrate gene-level structure with patient outcomes. Here, we present gANCHOR, a hierarchical hypergraph attention framework that integrates biologically informed representation learning with patient-level response prediction. By encoding gene-pathway relationships, gANCHOR learns pathway-aware cell embeddings that improve biological conservation and batch robustness across datasets. A cell-to-patient attention module aggregates cellular information to infer therapeutic response. Across benchmark datasets, gANCHOR achieved the strongest overall performance in biological conservation and batch-correction assessments. In response prediction across 161 patients, gANCHOR achieved the highest F1 score (0.87) among all compared foundation models. The model also identified reproducible gene signatures associated with response and non-response, providing interpretable biological insights.

<p align="center">  <img src="https://github.com/luoyuanlab/cart/blob/main/img_folder/Figure 1.png" height="800px" />  </p>  <br />

## **System requirements**
### **Hardware requirements**
gANCHOR requires a standard computer with optional GPU to support the in-memory operations.

### **OS Requirements**
The codes have been tested on the following systems:
* Linux: CentOS Linux 7
* Linux: Red Hat Enterprise Linux 8.10

### **Python Dependencies**
Expected installation time: ~15 minutes.
<pre>
anndata
math
matplotlib
numpy
torch
pandas
pickle
tqdm
umap
scanpy
scipy
seaborn
sklearn
</pre>

