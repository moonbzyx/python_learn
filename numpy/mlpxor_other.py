# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))

def s_prime(z):
    return np.multiply(z, 1.0-z)  # 修改的地方

def init_weights(layers, epsilon):
    weights = []
    for i in range(len(layers)-1):
        w = np.random.rand(layers[i+1], layers[i]+1)
        w = w * 2*epsilon - epsilon
        weights.append(np.mat(w))
    return weights

def fit(X, Y, w):
    # now each para has a grad equals to 0
    w_grad = ([np.mat(np.zeros(np.shape(w[i])))
              for i in range(len(w))])  # len(w) equals the layer number
    m, _ = X.shape
    h_total = np.zeros((m, 1))  # 所有样本的预测值, m*1, probability
    for i in range(m):
        x = X[i]
        y = Y[0,i]
        # forward propagate
        a = x
        a_s = []
        for j in range(len(w)):
            a = np.mat(np.append(1, a)).T
            a_s.append(a)  # 这里保存了前L-1层的a值
            z = w[j] * a
            a = sigmoid(z)
        h_total[i, 0] = a
        # back propagate
        delta = a - y.T
        w_grad[-1] += delta * a_s[-1].T  # L-1层的梯度
        # 倒过来，从倒数第二层开始到第二层结束，不包括第一层和最后一层
        for j in reversed(range(1, len(w))):
            delta = np.multiply(w[j].T*delta, s_prime(a_s[j]))  # 这里传递的参数是a，而不是z
            w_grad[j-1] += (delta[1:] * a_s[j-1].T)
    w_grad = [w_grad[i]/m for i in range(len(w))]
    J = (1.0 / m) * np.sum(-Y * np.log(h_total) - (np.array([[1]]) - Y) * np.log(1 - h_total))
    return {'w_grad': w_grad, 'J': J, 'h': h_total}


X = np.mat([[0,0],
            [0,1],
            [1,0],
            [1,1]])
Y = np.mat([0,1,1,0])
layers = [2,2,1]
epochs = 10
alpha = 0.5
w = init_weights(layers, 1)
result = {'J': [], 'h': []}
w_s = {}
for i in range(epochs):
    fit_result = fit(X, Y, w)
    w_grad = fit_result.get('w_grad')
    J = fit_result.get('J')
    h_current = fit_result.get('h')
    result['J'].append(J)
    result['h'].append(h_current)
    for j in range(len(w)):
        w[j] -= alpha * w_grad[j]
    if i == 0 or i == (epochs - 1):
        # print('w_grad', w_grad)
        w_s['w_' + str(i)] = w_grad[:]


# plt.plot(result.get('J'))
# plt.show()
print(w_s)
print(result.get('h')[0], result.get('h')[-1])
