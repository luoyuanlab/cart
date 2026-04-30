#  **gANCHOR: CAR T cell foundation model predicts immunotherapy response**

Single-cell transcriptomics provides unprecedented resolution for characterizing CAR T-cell states, yet translating heterogeneous cellular signals into accurate patient-level therapeutic response remains a major challenge. Existing studies primarily rely on experimental and statistical analyses to identify differentially expressed genes or cell populations associated with response, but lack predictive frameworks that systematically integrate gene-level structure with patient-level outcomes. Here, we present gANCHOR, a hierarchical hypergraph attention framework that integrates biologically informed representation learning with patient-level response prediction. By encoding gene-pathway relationships, gANCHOR learns pathway-aware cell embeddings that improve biological conservation and batch robustness across datasets. A cell-to-patient attention module aggregates cellular information to infer therapeutic response. Under a unified downstream framework, gANCHOR achieves superior prediction performance compared to existing foundation models. The model also identifies reproducible gene signatures associated with response and non-response, providing interpretable insights into relevant cellular programs.

<p align="center">  <img src="https://github.com/luoyuanlab/cart/blob/main/img_folder/Figure1.png" height="800px" />  </p>  <br />

## **System requirements**
### **Hardware requirements**
gANCHOR package requires a standard computer with optional GPU to support the in-memory operations.

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

