import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data
import matplotlib.pyplot as plt

param_niter = int(1e4)
param_delta = 0.1
param_lambda = 1e-4


class PTDeep(nn.Module):
    def __init__(self, config, activation_function=nn.ReLU()):
        '''
        Configurable deep model
        Args:
            config: list of integers, where each integer represents the number of neurons in a layer
                - first element is the number of input features
                - last element is the number of classes
            activation_function: activation function to use
        '''
        n_dim = config[0]
        n_classes = config[-1]
        super(PTDeep, self).__init__()
        self.layers = nn.Sequential()
        for i in range(1, len(config) - 1):
            self.layers.append(nn.Linear(config[i - 1], config[i]))
            self.layers.append(activation_function)
        self.layers.append(nn.Linear(config[-2], n_classes))

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        logits = self.layers(x)
        return logits
    
    def get_loss(self, X, Yoh_):
        loss = torch.mean(self.per_sample_loss(X, Yoh_))
        return loss
    
    def per_sample_loss(self, X, Yoh_):
        Yoh_ = torch.tensor(Yoh_, dtype=torch.float32)
        log_probs = nn.functional.log_softmax(self.forward(X), dim=1)
        loss = -torch.sum(Yoh_ * log_probs, dim=1)
        return loss
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

def train(model, X, Yoh_, param_niter, param_delta):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
    """
  
    # inicijalizacija optimizatora
    optimizer = optim.SGD(model.parameters(), lr=param_delta, weight_decay=param_lambda)

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
        - model: type: PTDeep
        - X: actual datapoints [NxD], type: np.array
        Returns: predicted class probabilites [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()
    X = torch.tensor(X, dtype=torch.float32)
    logits = model(X)
    probs = torch.softmax(logits, dim=1).detach().numpy()
    return probs


def ptdeep_decfun(model):
    def classify(X):
        probs = eval(model, X)
        if probs.shape[1] == 2:  # if classifying binary classes
            return probs[:, 1]
        return np.argmax(probs, axis=1)
    
    return classify
    

if __name__ == "__main__":
    configs = [[2, 2], [2, 10, 2], [2, 10, 10, 2]]

    # inicijaliziraj generatore slučajnih brojeva
    # np.random.seed(100)
    np.random.seed()

    X, Y_ = data.sample_gmm_2d(6, 3, 10)
    Yoh_ = data.class_to_onehot(Y_)
    ptdeep = PTDeep(config=[2,3])
    print(f'Config: [2,3]')
    print(f'Number of parameters: {ptdeep.count_params()}')
    train(ptdeep, X, Yoh_, param_niter=param_niter, param_delta=param_delta)
    probs = eval(ptdeep, X)
    Y = np.argmax(probs, axis=1)
    accuracy, pr, M = data.eval_perf_multi(Y, Y_)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {pr}')
    print(f'Matrix: {M}')

    decfun = ptdeep_decfun(ptdeep)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    data.graph_data(X, Y_, Y)
    plt.title('Config: [2,3]')
    plt.show()

    # definiraj model:
    for config in configs:
        X, Y_ = data.sample_gmm_2d(4, 2, 40)
        Yoh_ = data.class_to_onehot(Y_)

        ptdeep = PTDeep(config)
        print(f'Config: {config}')
        print(f'Number of parameters: {ptdeep.count_params()}')

        train(ptdeep, X, Yoh_, param_niter=param_niter, param_delta=param_delta)
        probs = eval(ptdeep, X)
        Y = np.argmax(probs, axis=1)
        accuracy, pr, M = data.eval_perf_multi(Y, Y_)
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {pr}')
        print(f'Matrix: {M}')

        decfun = ptdeep_decfun(ptdeep)
        bbox = (np.min(X, axis=0), np.max(X, axis=0))
        data.graph_surface(decfun, bbox, offset=0.5)
        data.graph_data(X, Y_, Y)
        plt.title(f'Config: {config}')
        plt.show()

        # instanciraj podatke X i labele Yoh_
        X, Y_ = data.sample_gmm_2d(6, 2, 10)
        Yoh_ = data.class_to_onehot(Y_)

        ptdeep = PTDeep(config)
        print(f'Config: {config}')
        print(f'Number of parameters: {ptdeep.count_params()}')

        train(ptdeep, X, Yoh_, param_niter=param_niter, param_delta=param_delta)
        probs = eval(ptdeep, X)
        Y = np.argmax(probs, axis=1)
        accuracy, pr, M = data.eval_perf_multi(Y, Y_)
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {pr}')
        print(f'Matrix: {M}')

        decfun = ptdeep_decfun(ptdeep)
        bbox = (np.min(X, axis=0), np.max(X, axis=0))
        data.graph_surface(decfun, bbox, offset=0.5)
        data.graph_data(X, Y_, Y)
        plt.title(f'Config: {config}')
        plt.show()
