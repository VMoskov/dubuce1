import numpy as np
import matplotlib.pyplot as plt
import data


param_niter = 425
param_delta = 0.01


def binlogreg_train(X, Y_):
    """
    Method that trains a binary logistic regression model
        Arguments:
            X: input data, np.array NxD
            Y_: class indices, np.array Nx1

        Returns:
            w: weights of the trained model
            b: bias of the trained model
    """

    w = np.random.randn(2)
    b = 0

    # gradient descent
    for i in range(param_niter):
        # classification measures
        scores = np.dot(X, w) + b  # Nx1

        # probabilities of class c_1
        probs = 1 / (1 + np.exp(-scores))  # Nx1

        # loss
        loss = -np.sum(np.log(probs)) / len(X)  # scalar

        # diagnostic output
        if i % 10 == 0:
            print(f"iteration {i}: loss {loss}")

        # derivative of loss with respect to scores
        dL_dscores = probs - Y_  # Nx1

        # gradient for weights and bias
        grad_w = np.dot(np.transpose(dL_dscores), X) / len(X)
        grad_b = np.sum(dL_dscores) / len(X)

        # update weights and bias
        w += -param_delta * grad_w
        b += -param_delta * grad_b

        if i % 15 == 0:
            decfun = binlogreg_decfun(w, b)
            bbox = (np.min(X, axis=0) - 1, np.max(X, axis=0) + 1)
            data.graph_surface(decfun, bbox, offset=0.5)

            Y = np.where(probs >= 0.5, 1, 0)
            data.graph_data(X, Y_, Y)

            plt.show()

    return w, b


def binlogreg_classify(X, w, b):
    """
    Method that classifies data points using logistical regression
        Arguments:
            X: input data, np.array NxD
            w: weights of the trained model
            b: bias of the trained model

        Returns:
            probs: class c1 probabilities for the data points
    """

    scores = np.dot(X, w) + b
    probs = 1 / (1 + np.exp(-scores))

    return probs


def binlogreg_decfun(w, b):
    def classify(X):
        return binlogreg_classify(X, w, b)
    return classify


if __name__ == "__main__":
    np.random.seed(100)
    # np.random.seed()

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 100)

    # train the model
    w, b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = np.where(probs >= 0.5, 1, 0)

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.argsort()])
    print(f"accuracy: {accuracy:.2f}, recall: {recall:.2f}, precision: {precision:.2f}, AP: {AP:.2f}")

    # graph the decision surface
    decfun = binlogreg_decfun(w, b)
    bbox = (np.min(X, axis=0)-1, np.max(X, axis=0)+1)
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y)

    # show the plot
    plt.show()
