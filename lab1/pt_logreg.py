import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data
import matplotlib.pyplot as plt


class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
        - D: dimensions of each datapoint 
        - C: number of classes
        """
        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        super(PTLogreg, self).__init__()
        self.W = nn.Parameter(torch.randn(D, C))
        self.b = nn.Parameter(torch.randn(C))

    def forward(self, X):
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        #   koristiti: torch.mm, torch.softmax
        X = torch.tensor(X, dtype=torch.float32)
        logits = torch.mm(X, self.W) + self.b
        return torch.softmax(logits, dim=1)

    def get_loss(self, X, Yoh_):
        # formulacija gubitka
        #   koristiti: torch.log, torch.exp, torch.sum
        #   pripaziti na numerički preljev i podljev
        Yoh_ = torch.tensor(Yoh_, dtype=torch.float32)
        probs = self.forward(X)
        log_probs = torch.log(probs)
        loss = torch.mean(-torch.sum(Yoh_ * log_probs, dim=1))
        return loss



def train(model, X, Yoh_, param_niter, param_delta):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
    """
  
     # inicijalizacija optimizatora
    optimizer = optim.SGD(model.parameters(), lr=param_delta)

    # petlja učenja
    for i in range(param_niter):
        optimizer.zero_grad()
        loss = model.get_loss(X, Yoh_)
        loss.backward()
        optimizer.step()

        # ispisujte gubitak tijekom učenja
        if i % 100 == 0:
            print(f'Iter: {i}, Loss: {loss.item()}')


def eval(model, X):
    """Arguments:
        - model: type: PTLogreg
        - X: actual datapoints [NxD], type: np.array
        Returns: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()
    X = torch.tensor(X, dtype=torch.float32)
    probs = model(X).detach().numpy()
    return probs


def ptlr_decfun(model):
    def classify(X):
        return np.argmax(eval(model, X), axis=1)
    return classify


if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    # np.random.seed(100)
    np.random.seed()


    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gmm_2d(K=6, C=3, N=10)
    Yoh_ = data.class_to_onehot(Y_)

    # definiraj model:
    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptlr, X, Yoh_, 1000, 0.5)

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptlr, X)

    # ispiši performansu (preciznost i odziv po razredima)
    Y = np.argmax(probs, axis=1)
    accuracy, pr, M = data.eval_perf_multi(Y, Y_)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {pr}')
    print(f'Matrix: {M}')

    # iscrtaj rezultate, decizijsku plohu
    decfun = ptlr_decfun(ptlr)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y)
    plt.show()