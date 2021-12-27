import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os

# os.makedirs(os.path.join('.', 'data'), exist_ok=True)
# file = os.path.join('.', 'data', 'Xor_W.npz')

LEARNINGRATE = 0.01

def sigmoid(x):
    x = np.array(x)
    return 1 / (1 + np.exp(-x))

def gradient_sigmoid(x):
    return np.multiply(sigmoid(x), 1-sigmoid(x))

def MSE(y,y_hat):
    return 0.5 * (y_hat - y) ** 2

def gradient(fun, x):
    h = 1e-5
    x = np.array(x)
    return (fun(x + h) - fun(x - h)) / (2.0 * h)

class Mlp_Xor():
    def __init__(self,is_training=True):
        self.z1 = np.zeros(2)
        self.a1 = np.zeros(2)
        self.z2 = 0.0
        self.y_hat = 0.0
        if is_training:
            self.w_1 = np.array(np.random.randn(6)).reshape(2,3) * 10
            self.w_2 = np.array(np.random.randn(3)) * 10
            # self.w_1 = np.array([-30,20,20,10,-20,-20]).reshape(2,3) 
            # self.w_2 = np.array([-10,20,20])
            np.savez('./data/Xor_W.npz', w1=self.w_1, w2=self.w_2)
            # print("Ok")
        else:
            data = np.load('./data/Xor_W.npz')
            self.w_1 = data["w1"]
            self.w_2 = data["w2"]

    def __call__(self,x):
        # x = np.append(1, x)
        self.z1 = np.dot(self.w_1, np.append(1, x))
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.w_2, np.append(1, self.a1))
        self.y_hat = sigmoid(self.z2)
        return self.y_hat

def gradient_W(net, x, y):

    # calculate the gradient for all steps
    d_L_y_hat = net.y_hat - y
    d_y_hat_z2 = gradient_sigmoid(net.z2)
    d_z2_w2 = np.append(1,net.a1)
    d_z2_a1 = np.array([net.w_2[1], net.w_2[2]])
    d_a1_z1 = np.array([[gradient_sigmoid(net.z1[1]), 0],
                        [0, gradient_sigmoid(net.z1[1])]])
    d_z1_w1 = np.array([[1, x[0], x[1], 0, 0, 0],
                        [0, 0, 0, 1, x[0], x[1]]])

    # backward, calculate the gradient of W1 and W2
    d_W2 = d_L_y_hat * d_y_hat_z2 * d_z2_w2
    d_W1 = d_L_y_hat * d_y_hat_z2 * d_z2_a1
    d_W1 = np.dot(d_W1, d_a1_z1)
    d_W1 = np.dot(d_W1, d_z1_w1).reshape(2,3)

    return d_W1, d_W2

def update_W(net,X,Y,lr=LEARNINGRATE):

    d1 = np.zeros(net.w_1.shape)
    d2 = np.zeros(net.w_2.shape)

    # forward propagation and BP to accumulate the gradient
    for j in range(4):
        net(X[j])
        dw1, dw2 = gradient_W(net,X[j],Y[j])
        # net.w_1 -= dw1
        # net.w_2 -= dw2
        d1 = d1 - dw1 * lr * 0.25
        d2 = d2 - dw2 * lr * 0.25
    # # update the parameters
    net.w_1, net.w_2 = net.w_1 - lr * d1, net.w_2 - lr * d2



def print_test(net, X, Y, before_update=True):
    if before_update:
        print("--------------The first test is-----------------")
    else:
        print("--------------After update it is-----------------")

    loss_data = []
    for i in range(4):
        loss_data.append(net(X[i]))
        if loss_data[-1] < 0.5:
            print(f"The pridict of Xor({X[i]}) is : 0  --  The net.y_hat is {loss_data[-1]}")
        else:
            print(f"The pridict of Xor({X[i]}) is : 1  --  The net.y_hat is {loss_data[-1]}")

    print(f"The net.w1 is :\n {net.w_1}")
    print(f"The net.w2 is :\n {net.w_2}\n")
    loss = [MSE(Y[i], loss_data[i]) for i in range(4)]
    # print(f"The loss is:{MSE(y,net.y_hat)}\n")
    print(f"loss is {np.mean(loss)}   and loss is {loss}")
    print("----------------------------------------------------")



if __name__ == '__main__':


    X = np.array([[0,0],[1,0],[0,1],[1,1]])
    # X = np.c_[np.ones(4),X]
    Y = np.array([0,1,1,0])
    # w_1 = np.array(np.random.randn(6)).reshape(2,3)



    # print(w_1)
    # print(w_2)
    # print(X)
    # print(Y)
    # print(sigmoid(np.array([0,1])))
    # print(np.sign(np.array([-0.5, 0, 0.9])))
    # print(w_1)
    # print(X[0])
    # print(np.dot(w_1, np.append(1,X[0])))
    # print(sigmoid(np.dot(w_1,X[0])))

    # print(gradient(sigmoid, [-1,0,0.5,3]))

    # print(z1)
    # net = Mlp_Xor()
    # print(net([0, 1]))
    # print(net([1, 1]))

    # print(sigmoid([-1, 0, 0.5]))
    # print(gradient_sigmoid([-1, 0, 0.5]))
    # print(gradient(sigmoid,[-1, 0, 0.5]))

    # net = Mlp_Xor()
    # y1 = net([1,0])
    # print(f"The y_hat is : {y1}")
    # net_w2 = net.w_2
    # print(f"The original w2 is: \n {net_w2}")
    # net.w_2 = np.array([0,0])
    # net_w2 = net.w_2
    # print(f"The modified w2 is: \n {net_w2}")
    # print(f"and the y_hat is {net([1,0])}")


    # data = np.load('./data/Xor_W.npz')
    # print(data['w1'])

    net = Mlp_Xor(is_training=True)


    # x, y = X[3],Y[3]


    print_test(net,X,Y)
    # update_W(net,X,Y,lr=0.1)
    # print_test(net,X,Y,before_update=False)

    lrr = 0.5

    for i in range(10000):
        update_W(net,X,Y,lr=lrr)
        if i % 1000 == 0:
            lrr -= 0.005
            print(f"loss is : {[MSE(Y[j],net(X[j])) for j in range(4)]}")

    print_test(net,X,Y,before_update=False)
