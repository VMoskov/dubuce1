import numpy as np
import matplotlib.pyplot as plt
import data


param_niter = int(1e5)
param_delta = 0.05
param_lambda = 1e-3
param_hdim = 5


sigmoid = lambda x: 1 / (1 + np.exp(-x))
relu = lambda x: np.maximum(0, x)
softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def fcann2_train(X, y):
    """
       Method that trains a probabilistic model for classification with one hidden layer
           Arguments:
               X: input data, np.array NxD
               y: class indices, np.array Nx1

           Returns:
               c: number of classes
               w: weights of the trained model
               b: bias of the trained model
       """
    # model layers: D -> 5 -> C, where D is input dimension (2), C is number of classes
    C = max(y) + 1
    D = X.shape[1]  # Input dimension

    # Initialize weights and biases
    W1 = np.random.randn(D, param_hdim)
    b1 = np.zeros(param_hdim)
    W2 = np.random.randn(param_hdim, C)
    b2 = np.zeros(C)

    # gradient descent
    for i in range(param_niter):
        s1 = X @ W1 + b1
        h1 = relu(s1)
        s2 = h1 @ W2 + b2

        probs = softmax(s2)
        log_probs = np.log(probs)

        loss = np.mean(-log_probs[np.arange(len(X)), y])  # extract the probabilities of the correct classes

        if i % 250 == 0:
            print(f"iteration {i}: loss {loss}")

        dL_ds2 = probs - np.eye(C)[y]
        dL_ds2 /= len(X)

        grad_W2 = h1.T @ dL_ds2 + param_lambda * W2
        grad_b2 = np.sum(dL_ds2, axis=0)

        dL_ds1 = dL_ds2 @ W2.T * (s1 > 0).astype(float)
        grad_W1 = X.T @ dL_ds1 + param_lambda * W1
        grad_b1 = np.sum(dL_ds1, axis=0)

        # Gradient update
        W1 -= param_delta * grad_W1
        b1 -= param_delta * grad_b1
        W2 -= param_delta * grad_W2
        b2 -= param_delta * grad_b2

    return W1, b1, W2, b2


def fcann2_classify(X, W1, b1, W2, b2):
    """
       Method that classifies data points using a probabilistic model for classification with one hidden layer
           Arguments:
               X: input data, np.array NxD
               w: weights of the trained model
               b: bias of the trained model

           Returns:
               probs: class probabilities for data points, np.array NxC
       """
    s1 = X @ W1 + b1
    h1 = relu(s1)
    s2 = h1 @ W2 + b2

    probs = softmax(s2)

    return probs


def fcann2_decfun(W1, b1, W2, b2):
    def classify(X):
        probs = fcann2_classify(X, W1, b1, W2, b2)
        return probs[:, 1]  # Extract only the probability of the positive class (class index 1)
    return classify


if __name__ == '__main__':
    # np.random.seed(100)
    np.random.seed()

    X, y = data.sample_gmm_2d(K=6, C=2, N=10)
    W1, b1, W2, b2 = fcann2_train(X, y)

    probs = fcann2_classify(X, W1, b1, W2, b2)
    y_pred = np.argmax(probs, axis=1)

    accuracy, pr, M = data.eval_perf_multi(y_pred, y)

    decfun = fcann2_decfun(W1, b1, W2, b2)
    bbox = (np.min(X, axis=0) - 1, np.max(X, axis=0) + 1)
    data.graph_surface(decfun, bbox, offset=0.5)

    data.graph_data(X, y, y_pred)
    plt.show()