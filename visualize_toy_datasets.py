import sys

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import train_test_split

from FeS_PCA import FeSK
from SPCA import SPCA
from data_utils import load_data

if __name__ == '__main__':

    Federated = True

    # obtain experiment index
    exp_idx = int(sys.argv[1]) - 3 if int(sys.argv[1]) >= 3 else int(sys.argv[1])
    kernel = 'rbf' if int(sys.argv[1]) > 2 else None
    dataset = ["xor", "rings", "iris"][exp_idx]

    #specify colours
    if dataset == "iris":
        colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
        cmap = LinearSegmentedColormap.from_list("iris-barshan", colors, N=3)
        show_data = False
    else:
        colors = [(0, 0, 1), (0, 1, 0)]
        cmap = LinearSegmentedColormap.from_list("toy-barshan", colors, N=2)
        show_data = False

    # load and scale data
    X, Y = load_data(dataset)
    lb = X.min()
    ub = X.max()
    X = (X - lb)/(ub-lb)

    # create train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42, stratify=Y)

    # one hot encode labels
    Y_train_1he = np.zeros((Y_train.size, Y.max() + 1))
    Y_train_1he[np.arange(Y_train.size), Y_train.reshape((-1,))] = 1

    if Federated:
        # obtain visually optimal gamma for current dataset
        gamma = {"xor":10, "rings":10,"iris":1000}
        X_args = {"gamma":gamma[dataset]}

        # create random orthogonal matrix masks
        P, _ = np.linalg.qr(np.random.randn( X_train.shape[1], X_train.shape[1]), mode='reduced')
        Q, _ = np.linalg.qr(np.random.randn( Y_train.shape[0], Y_train.shape[0]), mode='reduced')

        # obtain top 2 FeS(K)-PCA principal components using train data
        pca = FeSK(n_components=2, X_kernel=kernel, X_kernel_args=X_args, dual=False, secure_aggregation=False)

        X_support = pca.fit([X_train], [Y_train_1he], P, [Q])

    else:
        gamma = {"xor":7, "rings":10,"iris":1000}
        X_args = {"gamma":gamma[dataset]}
        pca = SPCA(n_components=2, X_kernel=kernel, X_kernel_args=X_args, dual=False)

        pca.fit([X_train], [Y_train_1he])

    # transform test and train data
    if kernel == None:
        if Federated:
            U_prime = pca.eigenvectors_
            U = np.dot(P.T, U_prime)
        else:
            U = pca.eigenvectors_
        Z_train = U.T.dot(X_train.T).T
        Z_test = U.T.dot(X_test.T).T
    else:
        U_prime = pca.eigenvectors_
        if Federated:
            K_train = pairwise_kernels(X_support, X_train, metric=pca.X_kernel, **pca.X_kernel_args)
            K_test = pairwise_kernels(X_support, X_test, metric=pca.X_kernel, **pca.X_kernel_args)
        else:
            K_train = pairwise_kernels(X_train, metric=pca.X_kernel, **pca.X_kernel_args)
            K_test = pairwise_kernels(X_train, X_test, metric=pca.X_kernel, **pca.X_kernel_args)
        
        Z_train = (U_prime.T).dot(K_train).T
        Z_test = (U_prime.T).dot(K_test).T

    #plot transformed test and train data
    plt.scatter(Z_train[:,1], Z_train[:,0], c=Y_train, cmap=cmap, marker=".")
    plt.scatter(Z_test[:,1], Z_test[:,0], c=Y_test, cmap=cmap, marker="+")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.close()