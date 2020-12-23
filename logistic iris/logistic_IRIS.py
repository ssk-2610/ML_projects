import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA
from multipledispatch import dispatch


def sigmoid(z):
    """
    The sigmoid function that returns the output of function for input z.
    This function will be used to predict the class for the given input vector.
    :param z:
    :return:
    """
    return 1 / (1 + np.exp(-z))


@dispatch(np.ndarray, np.ndarray, float, np.ndarray, int)
def gradient_descent(X, y, alpha, params, epoch):
    """
    This is an overloaded function is used to do gradient descend to find the optimal parameters for the given training
    dataset using the 'epoch' iterations. Here the iterations are taken to be 80.
    :param X: The feature data
    :param y: The classes for feature data
    :param alpha: Learning rate
    :param params: Initial parameters
    :param epoch: Number of iterations
    :return: a tuple of cost for every parameters used and optimal parameters
    """

    m = len(y)
    cost_history = np.zeros((epoch, 1))

    # Iterate for given number of iterations(80) the gradient descend
    for i in range(epoch):
        h_theta = sigmoid(X @ params)

        # Update the parameters and calculate the cost
        params = params - (alpha / m) * ((X.T) @ (h_theta - y))
        cost_history[i] = compute_cost_function(X, y, params)
    return (cost_history, params)


@dispatch(np.ndarray, np.ndarray, float, np.ndarray)
def gradient_descent(X, y, alpha, params):
    """
    This is an overloaded function is used to do gradient descend to find the optimal parameters for the given training
    dataset by iterating till we find the minima by gradient descend.
    :param X: The feature data
    :param y: The classes for feature data
    :param alpha: Learning rate
    :param params: Initial parameters
    :return: (cost_history, params): a tuple of cost for every parameters used and optimal parameters
    """

    m = len(y)
    epsilon = 0.01
    i = 0
    cost_history = np.zeros((16187, 1))

    # Iterate till we find the minima
    while i >= 0:
        h_theta = sigmoid(X @ params)
        j_value = (1 / m) * ((X.T) @ (h_theta - y))
        norm = LA.norm(j_value)

        # If norm is less than threshold then break because minima is reached
        if norm < epsilon:
            break

        # Update the parameters if threshold is not met and calculate the cost for new parameters
        params = params - (alpha * j_value)
        cost_history[i] = compute_cost_function(X, y, params)
        i += 1

    return (cost_history, params)


def compute_cost_function(X, y, params):
    """
    Computes the value of the cost function to be minimized
    :param X: The feature data
    :param y: The classes for feature data
    :param params: The calculated parametres
    :return: J_theta: Cost
    """

    m = len(y)
    h_theta = sigmoid(X @ params)
    epsilon = 1e-5

    # cost function
    J_theta = (1 / m) * (((-y).T @ np.log(h_theta + epsilon)) -
                         (1 - y).T @ np.log(1 - h_theta + epsilon))
    return J_theta


def predict(X, params):
    """
    predict the class using the features and parameters by giving input to sigmoid function
    :param X: The feature data
    :param params: Optimal parameters
    :return:0 or 1: predicted value of class
    """

    return np.round(sigmoid(X @ params))


def training(filename, option):
    """
    Base training program that parses the data and finds the parameters that can classify the data
    using the gradient descend method.
    :param filename: The name of training dataset file
    :param option: Type of gradient descend to be applied
    :return: params_optimal: The calculated parameters
    """

    # Parse the csv and change the first row to denote the features and class
    df = pd.read_csv(filename)
    df.rename(
        columns={'0': 'sepal length', '1': 'sepal width',
                 '2': 'petal length', '3': 'petal width', '4': 'species'},
        inplace=True)
    X = df[['sepal length', 'sepal width',
            'petal length', 'petal width']].to_numpy()
    y = []

    # Change the classes to 0 and 1
    for x in range(len(df)):
        if df.loc[x, 'species'] == 'Iris-versicolor':
            y.append(0.0)
        if df.loc[x, 'species'] == 'Iris-virginica':
            y.append(1.0)

    y = np.array(y)
    y = y.reshape((len(y), 1))

    m = len(y)
    n = np.size(X, 1)

    # Initialize the parameters with 0s
    params = np.zeros((n, 1))

    # Defined number of iterations and learning rate
    iterations = 80
    learning_rate = 0.02

    initial_cost = compute_cost_function(X, y, params)

    print("Initial Cost is: {} \n".format(initial_cost))

    # For different options chosen calculate the parameters accordingly
    # Option 1 number of iterations are not fixed and best parameters are calculated
    if option == 1:
        (cost_history, params_optimal) = gradient_descent(
            X, y, learning_rate, params)
        print("Optimal Parameters are: \n", params_optimal, "\n")

        plt.figure()
        sns.set_style('white')
        plt.plot(range(len(cost_history)), cost_history, 'r')
        plt.title("Convergence Graph of Cost Function")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost")
        plt.show()

        y_pred = predict(X, params_optimal)
        score = float(sum(y_pred == y)) / float(len(y))
        print("Accuracy on training data -> ", score)

    # For option 2 the number of iterations(epoch) are fixed = 80
    if option == 2:
        (cost_history, params_optimal) = gradient_descent(
            X, y, learning_rate, params, iterations)
        print("Optimal Parameters are: \n", params_optimal, "\n")

        plt.figure()
        sns.set_style('white')
        plt.plot(range(len(cost_history)), cost_history, 'r')
        plt.title("Convergence Graph of Cost Function")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost")
        plt.show()

        # Predict the class and check it the prediction is correct. Further calculate the
        # accuracy by total number of correct predictions / total number of predictions
        y_pred = predict(X, params_optimal)
        score = float(sum(y_pred == y)) / float(len(y))
        print("Accuracy on training data -> ", score)

    return params_optimal


def testing(testfile, params):
    """
    Parse the test dataset and check the calculated parameters found out from training to predict the classes
    and calculate the accuracy on the test dataset.
    :param testfile: Name of the test dataset file
    :param params: The optimal parameters calculated from training
    :return: None
    """

    # Parse the csv and change the first row to denote the features and class
    df = pd.read_csv(testfile)
    df.rename(
        columns={'0': 'sepal length', '1': 'sepal width', '2': 'petal length', '3': 'petal width', '4': 'species'},
        inplace=True)
    X = df[['sepal length', 'sepal width', 'petal length', 'petal width']].to_numpy()
    y = []

    # Change the classes to 0 and 1
    for x in range(len(df)):
        if df.loc[x, 'species'] == 'Iris-versicolor':
            y.append(0.0)
        if df.loc[x, 'species'] == 'Iris-virginica':
            y.append(1.0)

    y = np.array(y)
    y = y.reshape((len(y), 1))

    # Predict output class for the features in test dataset and accuracy score for the output
    output = predict(X, params)
    score = (output == y).mean()
    print("Accuracy on test data -> ", score)


def main():
    """
    Main driver function
    :return: None
    """

    train = input("Enter training data filename: ")
    test = input("Enter test data filename: ")
    option = int(input("Enter the option(1/2): \n1. For stopping criteria as ||∇J|| < ε\n2.For stopping criteria as "
                       "epochs= 80\n Choice => "))

    # Get the parameters from training the model on training dataset
    params = training(train, option)

    # Use the parameters to predict the class in test dataset and calculate the accuracy of the model
    testing(test, params)


if __name__ == '__main__':
    main()

