import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def accuracy_score(label,y_preds):
    ct=0
    for i in range(len(label)):
        if y_preds[i]==label[i]:
            ct=ct+1
    return ct/len(label)
data = pd.read_csv('training1.csv')

class NeuralNet:
    def __init__(self, x, y):
        np.random.seed(4)
        self.input = x
        self.w1 = np.random.randn(self.input.shape[1], 251)*0.01
        self.w2 = np.random.randn(251, 10)*0.01
        self.y = self.One_Hot_Encoder(y, x)
        self.output = np.zeros(self.y.shape)

    def feedataorward(self):
        x1 = self.sigmoid(np.dot(self.input, self.w1))
        x2 = self.sigmoid(np.dot(x1, self.w2))
        return x1, x2

    def backprop(self):
        for epoch in range(350):
            a1, a2 = self.feedataorward()
            d_w2 = 0.52 * \
                np.dot(
                    a1.T, ((1/self.input.shape[0])*(a2 - self.y) * self.sigmoid_der(a2)))
            d_w1 = 0.52*np.dot(self.input.T,  ((1/self.input.shape[0])*np.dot(
                (a2 - self.y) * self.sigmoid_der(a2), self.w2.T) * self.sigmoid_der(a1)))

            self.w1 -= d_w1
            self.w2 -= d_w2

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_der(self, x):
        vec = self.sigmoid(x)
        return vec*(1-vec)

    def One_Hot_Encoder(self, y, x):
        target = np.zeros((x.shape[0], 10))
        row, col = x.shape
        for i in range(row):
            target[i][y[i]] = 1
        return target


ydata = data.iloc[:, -1:]
xdata = data.iloc[:, : -1]

label = ydata.to_numpy()
x = xdata.to_numpy()


train_net = NeuralNet(x, label)
train_net.backprop()

a1, y_pred = train_net.feedataorward()
y_pred = np.argmax(y_pred, axis=1)


print("Training score {:.2f}%".format(accuracy_score(label, y_pred)*100))

data = pd.read_csv('test1.csv')
ydata = data.iloc[:, -1:]
xdata = data.iloc[:, : -1]

label = ydata.to_numpy()
x = xdata.to_numpy()

test_net = NeuralNet(x, label)
test_net.w1 = train_net.w1
test_net.w2 = train_net.w2
a1, y_pred = test_net.feedataorward()
y_pred = np.argmax(y_pred, axis=1)
print("Test score {:.2f}%".format(accuracy_score(label, y_pred)*100))
