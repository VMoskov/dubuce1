import numpy as np
import matplotlib.pyplot as plt
import data


param_niter = 425
param_delta = 0.01


def logreg_train(X, Y_):
    """
       Method that trains a binary logistic regression model
           Arguments:
               X: input data, np.array NxD
               Y_: class indices, np.array Nx1

           Returns:
               c: number of classes
               w: weights of the trained model
               b: bias of the trained model
       """

    # c = max(Y_) + 1
    c = 3
    w = np.random.randn(c, 2)
    b = np.zeros(c)

    # gradient descent
    for i in range(param_niter):
        # classification measures
        scores = np.dot(X, np.transpose(w)) + b  # NxC
        exp_scores = np.exp(scores)  # NxC

        # denominator of softmax function
        sum_exp_scores = np.sum(exp_scores, axis=1)  # Nx1

        # logartithmic class probabilities
        probs = exp_scores / sum_exp_scores[:, None]  # NxC
        log_probs = np.log(probs)  # NxC

        # loss
        loss = -np.sum(log_probs[np.arange(len(X)), Y_]) / len(X)  # scalar

        # diagnostic output
        if i % 10 == 0:
            print(f"iteration {i}: loss {loss}")

        # derivative of loss with respect to scores
        dL_dscores = probs - np.eye(c)[Y_]  # NxC

        # gradient for weights and bias
        grad_w = np.dot(np.transpose(dL_dscores), X) / len(X)
        grad_b = np.sum(dL_dscores, axis=0) / len(X)

        # update weights and bias
        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b


def logreg_classify(X, w, b):
    """
       Method that classifies data points using logistical regression
           Arguments:
               X: input data, np.array NxD
               w: weights of the trained model
               b: bias of the trained model

           Returns:
               probs: class probabilities for data points, np.array NxC
       """

    scores = np.dot(X, np.transpose(w)) + b  # NxC
    exp_scores = np.exp(scores)  # NxC

    # denominator of softmax function
    sum_exp_scores = np.sum(exp_scores, axis=1)  # Nx1

    # logartithmic class probabilities
    probs = exp_scores / sum_exp_scores[:, None]  # NxC

    return probs


def logreg_decfun(w, b):
    def classify(X):
        return np.argmax(logreg_classify(X, w, b), axis=1)
    return classify


if __name__ == "__main__":
    np.random.seed(100)
    # np.random.seed()

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(3, 100)

    # train the model
    w, b = logreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = logreg_classify(X, w, b)
    Y = np.argmax(probs, axis=1)

    # report performance
    accuracy, pr, M = data.eval_perf_multi(Y, Y_)
    print(f"accuracy: {accuracy}\nconfusion matrix:\n{M}\nprecision matrix: {pr}")

    # graph the decision surface
    decfun = logreg_decfun(w, b)
    bbox = (np.min(X, axis=0) - 1, np.max(X, axis=0) + 1)
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y)

    # show the results
    plt.show()
