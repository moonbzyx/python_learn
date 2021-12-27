import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def sigmoid(x):
    return 1.0 / (1.0+np.exp(-x))

def grad_sig_z(a): # sigmoid'(z) = a * (1 - a) 
    return np.multiply(a, 1.0-a) # a = sigmoid(z)

def mse(y,y_h):
    a = -np.multiply(y, np.log(y_h))
    b = -np.multiply((1-y), np.log(1.0-y_h))
    return a+b

def init_w(layers):
    w = []
    for i in range(len(layers) - 1):
        t = np.random.rand(layers[i+1], layers[i] +1)
        w.append( np.mat(t * 2.0 - 1.0) )  # return floats between -1 and 1
    return w

class mlp_xor():
    def __init__(self, layers, is_training=True):
        if is_training:
            self.w = init_w(layers)
        self.w_grad = [np.mat(np.zeros(self.w[l].shape)) 
                       for l in range(len(self.w))]

    def __call__(self, X,Y):
        Y_h = np.zeros(Y.shape)
        for i in range(X.shape[1]):
            # forward propagate
            a = X[:,i]  # every column of X
            a_stack_with_1= []
            for j in range(len(self.w)):
                a = np.row_stack((1,a))
                a_stack_with_1.append(a)
                z = self.w[j] * a
                a = sigmoid(z)
            Y_h[:,i] = a
            # backward propagate
            delta = a - Y[:,i]  # dL/da * da/dz
            self.w_grad[-1] += delta * a_stack_with_1[-1].T
            for j in reversed(range(1, len(self.w))):
                delta = np.multiply(self.w[j].T * delta, grad_sig_z(a_stack_with_1[j]))
                delta = delta[1:]
                self.w_grad[j-1] += (delta * a_stack_with_1[j-1].T)
        # "y += x_i and then y / n" is quite different from " y += x_i /  n" in computer 
        self.w_grad = [self.w_grad[j] / X.shape[1] for j in range(len(self.w))]
        return Y_h

def update_w(net, lr=0.5):
    for i in range(len(net.w)):
        net.w[i] -= lr * net.w_grad[i]

if __name__ == '__main__':

    X = np.mat([[0,0], [0,1], [1,0], [1,1]]).T
    Y = np.mat([0,1,1,0])
    layers = [2,2,1]
    net = mlp_xor(layers)

    fig = plt.figure()
    plt.ion()
    ax = plt.gca()
    ax.set_xlim([0,50])
    ax.set_ylim([0,1])
    st.title("The MLP of Xor")
    display_loss = st.empty()
    result_text = st.empty()
    x, y = [], []
    for i in range(5000):
        net(X,Y)
        update_w(net)
        if i % 100 == 0:
            x.append(i/100)
            y.append(mse(Y,net(X,Y)).mean())
            print("Loss is :", mse(Y,net(X,Y)).mean())
            ax.plot(x, y, 'b')
            display_loss.pyplot(fig)
            plt.pause(0.2)
    pred = np.mat([0 if net(X,Y)[:,i]<0.5 else 1 for i in range(4)]) 
    "The input data for xor X is :\n", [X]
    "The results Y is :\n", [Y]
    "The Prediction of the MLP net is : \n", [pred]
