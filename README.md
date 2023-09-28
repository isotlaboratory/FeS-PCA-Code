# FeS-PCA-Code
Repository containing a Python script that can be imported to evaluate FeS-PCA[1] and its variants. The unfederated counterparts are also available for comparison.

### Files:

`FeS_PCA.py`: Script containing FeSK class for applying FeS-PCA and its variants to distributed data.

`SPCA.py`: Script containing SPCA class for applying standard SPCA and its variants to centralized data.

`visualize_toy_datasets.py`: Script which applies either FeS-PCA or SPCA to toy datasets for recreating Fig. 6 from [1] and Fig. 1-a, 1-d, 4-a, 4-d, 5-a, 5-d, from [2] (Note recreation of figures from [2] is not exact since precise parameters for kernel function and dataset generation were not given).
    - First command line argument must be integer *x* in [0-5]: *x* mod 3 is dataset index in `['xor', 'rings', 'iris']` and ⌊*x*/2⌋ is method index in `['FeS-PCA/SPCA, FeSK-PCA/KSPCA]`.
    - Choose between applying federated or unfederated algorithms by setting variable `FEDERATED` on line 15 to `True` or `Flase`, respectively. 

`data_utils.py`: Script for generating toy datasets.

[1] W. Briguglio, W. A. Yousef, I. Traore, and M. Mamun, “Federated Supervised principal component analysis”, SUBMITTED to IEEE Transactions on Information Forensics and Security

[2] E. Barshan, A. Ghodsi, Z. Azimifar, and M. Z. Jahromi, “Supervised principal component analysis: Visualization, classification and regression on subspaces and submanifolds”, Pattern Recognition, vol. 44, no. 7, pp.
1357–1371, 2011.