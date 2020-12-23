#sk
import pandas as pd
import numpy as np

training_data = pd.read_csv('training1.csv', header=None)
test_data = pd.read_csv('test1.csv',  header=None)

y = np.transpose(training_data.values)[192]
test_y = np.transpose(test_data.values)[192]

n_features = 192
n_samples = 6669
n_classes = len(np.unique(y))


i = 0
X = np.zeros((n_samples, n_features))
X_test = np.zeros((3332, n_features))
for x in training_data.values:
    X[i] = x[:192]
    i = i+1

i = 0
for x in test_data.values:
    X_test[i] = x[:192]
    i = i+1


class Softmax:
    
    def __init__(self):
        self.W = None
        self.b = None
    
    def indicator_eq(self, x, y):
        if x == y:
            return 1
        else:
            return 0

    def softmax_func(self,z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def update_weights(self, X, y, n_features, n_samples, n_classes):
        z = X.dot(self.W) + self.b
        probs = self.softmax_func(z)
        entropy_loss = np.sum(np.log(probs))/n_samples
        dif_entropy = probs.copy()

        dif_entropy[np.arange(n_samples),y] -= 1
        dif_entropy /= n_samples

        delta_weight = X.T.dot(dif_entropy)
        delta_bias = np.sum(dif_entropy, axis=0, keepdims=True)

        return entropy_loss, delta_weight, delta_bias
    
    def train(self, X, y, learning_rate=1e-2, epochs=1000):
        n_features, n_samples = X.shape[1], X.shape[0]   
        n_classes = len(np.unique(y))

        if (self.W is None) & (self.b is None):
          np.random.seed(2020) 
          self.W = np.random.normal(loc=0.0, scale=1e-4, size=(n_features, n_classes))
          self.b = np.zeros((1, n_classes))

        for iter in range(epochs):
          loss, delta_weight, delta_bias = self.update_weights(X, y, n_features, n_samples, n_classes)

          self.W -= learning_rate*delta_weight
          self.b -= learning_rate*delta_bias
        
    def predict(self, X):
        y_pred = np.dot(X, self.W)+self.b
        y_pred = np.argmax(y_pred, axis=1)

        return y_pred

softmax = Softmax()
softmax.train(X, y, learning_rate=1e-2, epochs=1000)
print('Training accuracy '+ str(np.mean(softmax.predict(X)==y)*100)+" %")

print('Test accuracy '+ str(np.mean(softmax.predict(X_test)==test_y)*100)+" %")

