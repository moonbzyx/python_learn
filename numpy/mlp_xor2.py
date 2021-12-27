import numpy as np 

def sigmoid(x):
    x = np.array(x)
    return 1 / (1 + np.exp(-x))

def gradient_sigmoid(x):
    return np.multiply(sigmoid(x), 1-sigmoid(x))

def MSE(y,y_hat):
    return 0.5 * (y_hat - y) ** 2


class mlp_xor():
    def __init__(self,is_training=True):
        self.z1 = np.zeros(2)
        self.a = np.zeros(2)
        self.z2 = 0.0
        self.y_hat = 0.0
        self.b1=0.0
        self.b2=0.0
        if is_training:
            self.w1 = np.array(np.random.randn(4)).reshape(2,2)
            self.w2 = np.array(np.random.randn(2))
            # self.w1 = np.array([0.1,2,3,0.4]).reshape(2,2)
            # self.w2 = np.array([0.1,9])
            np.savez('./data/xor2_w.npz', w1=self.w1, w2=self.w2)
            # print("Ok")
        else:
            data = np.load('./data/xor2_w.npz')
            self.w1 = data["w1"]
            self.w2 = data["w2"]

    def __call__(self,x):
        self.z1 = np.dot(self.w1, x) + self.b1
        self.a = sigmoid(self.z1)
        self.z2 = np.dot(self.w2, self.a) + self.b2
        self.y_hat = sigmoid(self.z2)
        return self.y_hat

def update_w(net, x, y,lr = 0.1):
    dL_yhat = net.y_hat - y
    dYhat_z2 = gradient_sigmoid(net.z2)
    dZ2_w2 = np.array(net.a)
    dZ2_b2 = 1
    dZ2_a = net.w2
    dA_z1 = np.array([[gradient_sigmoid(net.z1[0]), 0],
                      [0, gradient_sigmoid(net.z1[1])]])
    dZ1_w1 = np.array([[x[0], x[1], 0, 0],
                       [0, 0, x[0], x[1]]])
    dZ1_b1 = np.array([1, 1])

    dL_w2 = dL_yhat * dYhat_z2 * dZ2_w2
    dL_b2 = dL_yhat * dYhat_z2 * dZ2_b2
    tem = dL_yhat * dYhat_z2 * dZ2_a   
    tem = np.dot(tem, dA_z1)
    dL_w1 = np.dot(tem, dZ1_w1).reshape(2,2)
    dL_b1 = np.dot(tem, dZ1_b1)

    net.w1 -= lr * dL_w1
    net.w2 -= lr * dL_w2
    net.b1 -= lr * dL_b1
    net.b2 -= lr * dL_b2

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

    print(f"The net.w1 is :\n {net.w1}")
    print(f"The net.w2 is :\n {net.w2}\n")
    print(f"The net.b1 is :{net.b1},  The net.b2 is :{net.b2}\n")
    loss = [MSE(Y[i], loss_data[i]) for i in range(4)]
    # print(f"The loss is:{MSE(y,net.y_hat)}\n")
    print(f"loss is {np.mean(loss)}   and loss is {loss}")
    print("----------------------------------------------------")


if __name__ == '__main__':

    X = np.array([[0,0],[1,0],[0,1],[1,1]])
    Y = np.array([0,1,1,0])
    net = mlp_xor()
    print_test(net,X,Y)
    for i in range(1000):
        for j in range(4):
            update_w(net,X[j],Y[j],lr=0.1)
        if i % 100 == 0:
            loss = [MSE(Y[k],net(X[k])) for k in range(4)]
            print(f"loss is {np.mean(loss)}   and loss is {loss}")
    print_test(net,X,Y,before_update=False)


