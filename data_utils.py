import numpy as np

def load_data(dataset, n=None, m=1000):
    """
        args:
            dataset(str): specifies which dataset to load
            n(int): number of samples to generate, only used with dataset="synthetic"
        returns:
            X(numpy array): nxm array of samples
            Y(numpy array): nx1 array of labels (not returned if dataset="synthetic")
    """


    if dataset == "iris":
        from sklearn.datasets import load_iris

        data = load_iris()
        X = data["data"]
        Y = data["target"]

    elif dataset == "xor":
        from scipy.stats import multivariate_normal

        n = 400
        spc = int(n/4)
        cov = 0.4
        A = 0
        B = 5

        y, mean = (0,(A,A))
        X = multivariate_normal.rvs(mean=mean, cov=cov, size=spc)
        Y = [y] * spc
        for y, mean in [(1,(A,B)), (1,(B,A)),(0,(B,B))]:
            X = np.append(X, multivariate_normal.rvs(mean=mean, cov=cov, size=spc), axis=0)
            Y += [y] * spc
        Y = np.array(Y)

        noise = np.random.normal(0,1,n)
        X[:,0] = X[:,0]+noise
        X[:,1] = X[:,1]+noise

    elif dataset == "rings":

        Y = np.zeros((317,1)).astype(int)
        X = np.zeros((317,2))
        k = 0
        for i in range(0,21):
            for j in range(0,21):
                x0 = i/20
                x1 = j/20

                d = np.sqrt((0.50 - x0)**2 + (0.50 - x1)**2)

                if d <= 0.25:
                    X[k,0] = x0
                    X[k,1] = x1
                    Y[k] = 0

                    k+=1
                elif d <= 0.50:
                    X[k,0] = x0
                    X[k,1] = x1
                    Y[k] = 1

                    k+=1
    else:
        raise Exception('Dataset "'+dataset+'" not recognized.')

    Y = Y.reshape((-1,1)).astype(np.int)

    return X, Y