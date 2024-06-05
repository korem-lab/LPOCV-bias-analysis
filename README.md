# Distributional bias compromises leave-one-out cross-validation

This repo contains all code supporting the analysis for our manuscript "Distributional bias compromises leave-one-out cross-validation".


Outline of the Repository
------------------
| Folder/file | Description |
|--|--|
| `Simulations.py` | This script runs all of the simulation analyses referenced in our mansucript .|
| `CFS` | Contains the code used to replicated the original analysis from the CFS analysis by Vogl et al., (https://www.science.org/doi/10.1126/sciadv.abq2422), and with our additional RLOOCV implementation. All data used within this folder is available under the original publication, along with the code used to replicate the original work's findings |
| `PTB` | Contains the code used to benchmark LOOCV and RLOOCV linear models on data from Fettweiss et al. (https://www.nature.com/articles/s41591-019-0450-2), using the publicly available processed data from Huang et al. (https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-023-01702-2) |
| `ICI-CD4` | Contains the code used to replicated the original analysis from the ICI-CD4 analysis by Lozano et al., (https://www.nature.com/articles/s41591-021-01623-z), and with our additional RLOOCV implementation. All data used within this folder is available under the original publication,  along with the code used to replicate the original work's findings. |
| delong.py | code to run the delong test, this file was obtained from `https://github.com/yandexdataschool/roc_comparison`|
| plots-latest | This folder contains all images, figures, and tables referenced within our manuscript. All code executed from all analyses write outputs into this folder |
