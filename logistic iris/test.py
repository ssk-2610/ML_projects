import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA
from multipledispatch import dispatch


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@dispatch(np.ndarray, np.ndarray, float, np.ndarray, int)
def gradient_descent(X, y, alpha, params, epoch):
    m = len(y)
    cost_history = np.zeros((epoch, 1))
    for i in range(epoch):
        h_theta = sigmoid(X @ params)
        params = params - (alpha / m) * ((X.T) @ (h_theta - y))
        cost_history[i] = compute_cost_function(X, y, params)
    return (cost_history, params)


@dispatch(np.ndarray, np.ndarray, float, np.ndarray)
def gradient_descent(X, y, alpha, params):
    m = len(y)
    epsilon = 0.01
    i = 0
    cost_history = np.zeros((16187, 1))
    while i >= 0:
        h_theta = sigmoid(X @ params)
        j_value = (1 / m) * ((X.T) @ (h_theta - y))
        norm = LA.norm(j_value)
        if norm < epsilon:
            break
        params = params - (alpha * j_value)
        cost_history[i] = compute_cost_function(X, y, params)
        i += 1

    return (cost_history, params)


def compute_cost_function(X, y, params):
    m = len(y)
    h_theta = sigmoid(X @ params)
    epsilon = 1e-5
    J_theta = (1 / m) * (((-y).T @ np.log(h_theta + epsilon)) - (1 - y).T @ np.log(1 - h_theta + epsilon))
    return J_theta


def predict(X, params):
    return np.round(sigmoid(X @ params))


def training(filename, option):
    df = pd.read_csv(filename)
    df.rename(
        columns={'0': 'sepal length', '1': 'sepal width', '2': 'petal length', '3': 'petal width', '4': 'species'},
        inplace=True)
    X = df[['sepal length', 'sepal width', 'petal length', 'petal width']].to_numpy()
    y = []
    for x in range(len(df)):
        if df.loc[x, 'species'] == 'Iris-versicolor':
            y.append(0.0)
        if df.loc[x, 'species'] == 'Iris-virginica':
            y.append(1.0)

    y = np.array(y)
    y = y.reshape((len(y), 1))

    m = len(y)

    n = np.size(X, 1)
    params = np.zeros((n, 1))

    iterations = 80
    learning_rate = 0.02

    initial_cost = compute_cost_function(X, y, params)

    print("Initial Cost is: {} \n".format(initial_cost))
    if option == 1:
        (cost_history, params_optimal) = gradient_descent(X, y, learning_rate, params)
        print("Optimal Parameters are: \n", params_optimal, "\n")

        plt.figure()
        sns.set_style('white')
        plt.plot(range(len(cost_history)), cost_history, 'r')
        plt.title("Convergence Graph of Cost Function")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost")
        plt.show()
        return params_optimal

    if option == 2:
        (cost_history, params_optimal) = gradient_descent(X, y, learning_rate, params, iterations)
        print("Optimal Parameters are: \n", params_optimal, "\n")

        plt.figure()
        sns.set_style('white')
        plt.plot(range(len(cost_history)), cost_history, 'r')
        plt.title("Convergence Graph of Cost Function")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost")
        plt.show()
        return params_optimal


def scoreontest(testfile, params):
    df = pd.read_csv(testfile)
    df.rename(
        columns={'0': 'sepal length', '1': 'sepal width', '2': 'petal length', '3': 'petal width', '4': 'species'},
        inplace=True)
    X = df[['sepal length', 'sepal width', 'petal length', 'petal width']].to_numpy()
    y = []
    for x in range(len(df)):
        if df.loc[x, 'species'] == 'Iris-versicolor':
            y.append(0.0)
        if df.loc[x, 'species'] == 'Iris-virginica':
            y.append(1.0)

    y = np.array(y)
    y = y.reshape((len(y), 1))

    output = []
    score = 0
    output = predict(X, params)

    score = (output == y).mean()
    print("Accuracy -> ", score)


train = input("Enter training data file: ")
test = input("Enter test data file: ")
option = int(input("Enter the option(1/2): \n1. For stopping criteria as ||∇J|| < ε\n2.For stopping criteria as "
                   "epochs= 80\n Choice => "))
params = training(train, option)
scoreontest(test, params)
