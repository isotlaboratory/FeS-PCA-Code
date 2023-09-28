# FeS-PCA-Code
Repository containing a Python script that can be imported to evaluate FeS-PCA and its variants. The unfederated counterparts are also available for comparison.

#### Files:

`FeS_PCA.py`: Script containing FeSK class for applying FeS-PCA and its variants to distributed data.

`SPCA.py`: Script containing SPCA class for applying standard SPCA and its variants to centralized data.

`visualize_toy_datasets.py`: Script which applies either FeS-PCA or SPCA to toy datasets for recreating Fig. 6 from [1] and Fig. 1-a, 1-d, 4-a, 4-d, 5-a, 5-d, from [2] (Note recreation of figures from [2] is not exact since precise parameters for kernel function and dataset generation were not given).

`data_utils.py`: Script for generating toy datasets.