import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import pairwise_kernels

def inplace_sum(arrlist):
    sum = arrlist[0].copy()
    for a in arrlist[1:]:
        sum += a
    return sum

def secure_aggregation(xs):
    #code ref https://github.com/Di-Chai/FedSVD/blob/df83dadea5a910106066bb008e4b46d55a05fe80/utils.py#L23

    n = len(xs)
    size = xs[0].shape
    # Step 1 Generate random samples between each other
    perturbations = []
    for i in range(n):
        tmp = []
        for j in range(n):
            tmp.append(np.random.randint(low=-10**5, high=10**5, size=size) + np.random.random(size))
        perturbations.append(tmp)
    perturbations = np.array(perturbations)
    perturbations -= np.transpose(perturbations, [1, 0, 2, 3])
    ys = [xs[i] - np.sum(perturbations[i], axis=0) for i in range(n)]
    results = np.sum(ys, axis=0)

    return results

class FeSK:
    
    def __init__(self, n_components=None, X_kernel=None, X_kernel_args={}, Y_kernel=None, Y_kernel_args={}, dual=False, secure_aggregation=True, K_centroids=50, eps=1e-10):
        """
            n_components(int): The number of principal components to use for projection.
            X_kernel(str): The kernel used to generate the data kernel matrix used in FeSK-PCA, generated by sklearn.metrics.pairwise.
            X_kernel_args(dictionary): Arguments passed to kernel function specified by X_kernel.
            Y_kernel(str): The kernel used to generate the label kernel matrix, only applicable to dual and FeSK-PCA, generated by sklearn.metrics.pairwise.
            Y_kernel_args(dictionary): Arguments passed to kernel function specified by Y_kernel.
            dual(boolean): Weather to use the dual formulation of FeS-PCA (ignored if X_kernel != None).
            secure_aggregation(boolean): weather to use secure aggregation when collecting X_mean, PXQ, or PHX. (ignored if X_kernel != None).
            K_centroids(int): Total number of centroids to use in X_support. (ignored if X_kernel == None).
            eps(float): Very small number added to the diagonal of L in dual FeS-PCA or K_support in FeSK-PCA for numerical stability
        """

        self.n_components_ = n_components
        self.X_kernel = X_kernel
        self.X_kernel_args = X_kernel_args
        self.Y_kernel = Y_kernel
        self.Y_kernel_args = Y_kernel_args
        self.dual = dual
        self.secure_aggregation = secure_aggregation
        self.K_centroids = K_centroids
        self.eps = eps

    def fit(self, Xs, Ys, P=None, Qs=None):
        """
        Xs(list of numpy arrays): each clients nxm feature matrix.
        Ys(list of numpy arrays): each clients corresponding label vector, if concatenated along axis 0, result should be nxc.
        P(mxm numpy array): shared random orthogonal matrix not known to server (Ignored if self.X_kernel == None).
        Qs(list of numpy arrays): each client's corresponding random orthogonal matrix only known to them, if concatenated along axis 0, result should be nxn (Ignored if self.X_kernel == None).

        ref: Barshan et al. "Supervised Principal Component Analysis: Visualization, Classification and Regression on Subspaces and Submanifolds" 2011
        """

        n = sum([cur_X.shape[0] for cur_X in Xs])
        m = Xs[0].shape[1]
        c = Ys[0].shape[1]

        if self.dual == True and self.X_kernel == None: #dual SPCA
            
            # compute centring matrix
            e = np.ones((n,1))
            I = np.identity(n)
            H = I - (1/n) * np.dot(e,e.T)

            # clients mask their data
            Xs_masked = []
            for k in range(len(Xs)):
                Xs_masked.append(P @ (Xs[k].T))

            # concatenate masked client data and multiply by centring matrix
            PXH = np.concatenate(Xs_masked, axis=1).dot(H)

            # concatenate plain text labels
            Y = np.concatenate(Ys,axis=0)

            # server computes PXHDelta
            if self.Y_kernel == 'rbf':
                L = pairwise_kernels(Y.reshape(-1, 1), metric=self.Y_kernel, **self.Y_kernel_args)
            else:
                L = Y.dot(Y.T)
            Delta = np.linalg.cholesky(L+np.identity(Y.shape[0]))
            psi = PXH.dot(Delta)

            # determine index of top principal components
            m = Xs[0].shape[0]
            if self.n_components_ != None:
                indexes=[m-self.n_components_, m-1]
            else:
                indexes=None

            # U, the eigenvectors, correspond to V from Section 5.2 of Barshan et al.
            V, U = sp.linalg.eigh((psi.T).dot(psi), subset_by_index=indexes)

            # V, the eigenvalues, used to create Sigma from Section 5.2 of Barshan et al.
            Sigma = np.sqrt(V*np.identity(V.shape[0]))
            Sigma_inv = np.linalg.inv(Sigma)

            U = psi.dot(U).dot(Sigma_inv)

            self.eigenvectors_ = U #eigen vectors
            self.eigenvalues_ = V #eigen values

        elif self.X_kernel == None: #standard SPCA

            X_means = []
            for i in range(len(Xs)): # clients compute the means of their data
                X_means.append(np.mean(Xs[i], axis=0).reshape((1,-1))*(Xs[i].shape[0]))

            if self.secure_aggregation: # securely average all client's mean
                X_mean = secure_aggregation(X_means)[0]/n
            else: # average all client's means
                X_mean = np.sum(X_means, axis=0)[0]/n    

            # each client mean centres their data
            for i in range(len(Xs)):
                Xs[i] = Xs[i] - X_mean

            Xs_masked = []
            Ys_masked = []
            for k in range(len(Xs)): # each client masks their data and labels
                Xs_masked.append(P @ (Xs[k].T) @ Qs[k])
                Ys_masked.append(Qs[k].T @ Ys[k])

            # securely sum masked matrices
            if self.secure_aggregation:
                PXQ = secure_aggregation(Xs_masked)
                QtY = secure_aggregation(Ys_masked)
            else:
                PXQ = inplace_sum(Xs_masked)
                QtY = inplace_sum(Ys_masked)

            # determine index of top principal components
            m = Xs[0].shape[1]
            if self.n_components_ != None:
               indexes=[m-self.n_components_, m-1]
            else: 
               indexes=None

            # server computes supervised principal components
            Q = PXQ.dot(QtY.dot(QtY.T)+np.identity(QtY.shape[0])).dot(PXQ.T)
            V, U = sp.linalg.eigh(Q, subset_by_index=indexes)
            
            self.eigenvectors_ = U #eigen vectors
            self.eigenvalues_ = V #eigen valuess
            
        else: # Kernel SPCA

            # determine each client's share of centroids
            samples_per_client = np.array([cur_X.shape[0] for cur_X in Xs])/n
            centroids_per_client = np.floor(self.K_centroids * samples_per_client).astype(int)


            Y_support = []
            X_support = []
            remainder = self.K_centroids - np.sum(centroids_per_client)
            for i in range(len(Xs)):
                # at client i

                cur_centroids = centroids_per_client[i]
                if i < remainder:
                    cur_centroids+=1

                # obtain centroids
                kmeans = KMeans(n_clusters=cur_centroids, random_state=0, n_init=10).fit(Xs[i], Ys[i])
                X_support.append(kmeans.cluster_centers_)

                # classify centroids from current client using client's data
                n_neighbors=5
                neigh = KNeighborsClassifier(n_neighbors=n_neighbors, p=1, n_jobs=-1)
                neigh.fit(Xs[i], Ys[i])
                Y_support.append(neigh.predict(X_support[-1]))

            Y_support = np.concatenate(Y_support).reshape(-1,c)
            X_support = np.concatenate(X_support)

            # at server, obtain kernel matrix from centroids
            K_support = pairwise_kernels(X_support, metric=self.X_kernel, **self.X_kernel_args)
            
            # calculate centering matrix
            e = np.ones((K_support.shape[0],1))
            I = np.identity(K_support.shape[0])
            H = I - (1/K_support.shape[0]) * np.dot(e,e.T)

            # determine index of top principal components
            if self.n_components_ != None:
                indexes=[ K_support.shape[0]-self.n_components_, K_support.shape[0]-1]
            else:
                indexes=None

            if self.Y_kernel == 'rbf':
                L = pairwise_kernels(Y_support.reshape(-1, 1), metric=self.Y_kernel, **self.Y_kernel_args)
            else:
                L = Y_support.dot(Y_support.T)

            # server computes supervised principal components
            Q = K_support.dot(H).dot(L+np.identity(L.shape[0])).dot(H).dot(K_support)
            K_support = K_support + (self.eps*np.identity(K_support.shape[0]))
            # U, the eigenvectors, corresponds to β in Section 5.3.2 of Barshan et al.
            V, U = sp.linalg.eigh(Q, b=K_support, subset_by_index=indexes)
            
            self.eigenvectors_ = U # eigen vectors
            self.eigenvalues_ = V # eigen values

            return X_support

        return -1