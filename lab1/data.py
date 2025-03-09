import numpy as np
import matplotlib.pyplot as plt


class Random2DGaussian:
    """
    Class that represents a 2D Gaussian distribution with random parameters

    Attributes:
        min_x: minimum value of x
        max_x: maximum value of x
        min_y: minimum value of y
        max_y: maximum value of y

    Methods:
        constructor: creates a random 2D Gaussian distribution
            span_x, span_y: span of the distribution in x and y
            mean: mean of the distribution
            eigval_x, eigval_y: eigenvalues of the distribution
            D: diagonal matrix of the eigenvalues
            theta: rotation angle of the distribution
            R: rotation matrix of the distribution
            sigma: covariance matrix of the distribution
        get_sample(n): returns n samples from the distribution
    """

    min_x = 0
    max_x = 10
    min_y = 0
    max_y = 10

    def __init__(self):
        self.span_x, self.span_y = self.max_x - self.min_x, self.max_y - self.min_y
        self.mean = np.random.sample(2) * (self.span_x, self.span_y) + (self.min_x, self.min_y)
        self.eigval_x, self.eigval_y = (np.random.sample(2) * (self.span_x, self.span_y) / 5)**2
        self.D = np.diag([self.eigval_x, self.eigval_y])
        self.theta = 2 * np.random.sample() * np.pi
        self.R = np.array([
            [np.cos(self.theta), -np.sin(self.theta)],
            [np.sin(self.theta), np.cos(self.theta)]])
        self.sigma = np.dot(np.dot(np.transpose(self.R), self.D), self.R)

    def get_sample(self, n):
        return np.random.multivariate_normal(self.mean, self.sigma, n)


def sample_gauss_2d(C, N):
    X = np.vstack([d.get_sample(N) for d in [Random2DGaussian() for i in range(C)]])
    Y_ = np.hstack([[i] * N for i in range(C)])
    return X, Y_


def sample_gmm_2d(K, C, N):
    G = [Random2DGaussian() for _ in range(K)]  # K Gaussian components
    Y = np.random.randint(0, C, K)
    X = np.vstack([G_i.get_sample(N) for G_i in G])  # N samples from each component
    Y_ = np.hstack([[i] * N for i in Y])  # N samples from each component
    return X, Y_


def eval_perf_binary(Y, Y_):
    TP = np.sum((Y == 1) & (Y_ == 1))  # true positives
    FP = np.sum((Y == 1) & (Y_ == 0))  # false positives
    TN = np.sum((Y == 0) & (Y_ == 0))  # true negatives
    FN = np.sum((Y == 0) & (Y_ == 1))  # false negatives

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)

    return accuracy, recall, precision


def eval_perf_multi(Y, Y_):
    precision_matrix = []
    c = max(Y_) + 1
    M = np.zeros((c, c))
    for true_class, predicted_class in zip(Y_, Y):
        M[true_class][predicted_class] += 1

    for i in range(c):
        TP = M[i][i]
        FP = np.sum(M[:, i]) - TP
        FN = np.sum(M[i, :]) - TP
        TN = np.sum(M) - TP - FP - FN
        recall = TP / (TP + FN) if TP + FN > 0 else 0.0
        precision = TP / (TP + FP) if TP + FP > 0 else 0.0
        precision_matrix.append((recall, precision))

    accuracy = np.trace(M) / np.sum(M)
    return accuracy, precision_matrix, M


def eval_AP(Y_):
    N = len(Y_)
    pos = np.sum(Y_)
    neg = N - pos

    TP = pos
    FP = neg

    precision_sum = 0
    for Y_i in Y_:
        precision = TP / (TP + FP)
        if Y_i:
            precision_sum += precision
        TP -= Y_i
        FP -= not Y_i
    return precision_sum / pos if pos > 0 else 0


def graph_data(X, Y_, Y, special=[]):
    """
    Plot the data points in X, color-coded according to the class in Y
        Arguments:
            X: data points
            Y_: true class
            Y: predicted class
    """
    colors = np.where(Y_ == 0, "grey", np.where(Y_ == 1, "white", "darkslategray"))
    sizes = np.repeat(20, len(Y_))
    sizes[special] = 50
    correct = Y == Y_
    wrong = Y != Y_

    plt.scatter(X[correct, 0], X[correct, 1], marker='o', c=colors[correct], edgecolors='k')
    plt.scatter(X[wrong, 0], X[wrong, 1], marker='s', c=colors[wrong], edgecolors='k')


def graph_surface(function, rect, offset=0.5, width=256, height=256):
    """Creates a surface plot (visualize with plt.show)

    Arguments:
      function: surface to be plotted
      rect:     function domain provided as:
                ([x_min,y_min], [x_max,y_max])
      offset:   the level plotted as a contour plot

    Returns:
      None
    """

    lsw = np.linspace(rect[0][1], rect[1][1], width)
    lsh = np.linspace(rect[0][0], rect[1][0], height)
    xx0, xx1 = np.meshgrid(lsh, lsw)
    grid = np.stack((xx0.flatten(), xx1.flatten()), axis=1)

    # get the values and reshape them
    values = function(grid).reshape((width, height))

    # fix the range and offset
    delta = offset if offset else 0
    maxval = max(np.max(values) - delta, - (np.min(values) - delta))

    # draw the surface and the offset
    plt.pcolormesh(xx0, xx1, values,
                   vmin=delta - maxval, vmax=delta + maxval)

    if offset is not None:
        plt.contour(xx0, xx1, values, colors='black', levels=[offset])


def myDummyDecision(X):
    scores = X[:, 0] + X[:, 1] - 5
    return scores


def class_to_onehot(Y):
    C = max(Y) + 1
    Yoh_ = np.zeros((len(Y), C))
    Yoh_[range(len(Y)), Y] = 1
    return Yoh_


if __name__ == "__main__":
    # np.random.seed(100)
    np.random.seed()

    # get the training dataset
    X, Y_ = sample_gmm_2d(K=4, C=2, N=30)
    print(X.shape, Y_.shape)

    # get the class predictions
    Y = myDummyDecision(X) > 0.5

    # graph the decision surface
    rect = (np.min(X, axis=0)-1, np.max(X, axis=0)+1)
    graph_surface(myDummyDecision, rect, offset=0)

    # graph the data points
    graph_data(X, Y_, Y)

    # show the results
    plt.show()
