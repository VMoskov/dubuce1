import numpy as np
from sklearn import svm
import data
import matplotlib.pyplot as plt
import pt_deep
import torch


class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1.0, param_svm_gamma='auto'):
        self.svm = svm.SVC(C=param_svm_c, gamma=param_svm_gamma)
        self.svm.fit(X, Y_)

    def predict(self, X):
        return self.svm.predict(X)

    def get_scores(self, X):
        return self.svm.decision_function(X)
    
    def support(self):
        return self.svm.support_
    

if __name__ == '__main__':
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X,Y_ = data.sample_gmm_2d(6, 2, 10)

    
    svm_model = KSVMWrap(X, Y_)
    Y = svm_model.predict(X)

    # iscrtaj rezultate, decizijsku plohu
    # graph the decision surface
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    decfun = svm_model.get_scores
    data.graph_surface(decfun, bbox, offset=0)
    
    # graph the data points
    data.graph_data(X, Y_, Y, special=svm_model.support())
    plt.show()