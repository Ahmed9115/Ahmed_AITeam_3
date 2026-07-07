import numpy as np

class logistic_regression :

    def __init__(self ,iterations,lr):
        self.iterations = iterations
        self.lr = lr

    def fit(self , x_train ,y_train):
        x_train = np.hstack([np.ones((x_train.shape[0],1)) , x_train])
        weights = np.zeros((x_train.shape[1],1)) + 0.5 
        for iter in range(self.iterations):
            z = x_train@weights
            h = 1 / (1 + np.exp(-z))
            weights -= (self.lr/len(x_train))*(x_train.T @ (h - y_train))
        self.weights = weights

    def predict(self ,x_test):
        x_test = np.hstack([np.ones((x_test.shape[0],1)) , x_test])
        z = x_test@(self.weights)
        h = 1 / (1 + np.exp(-z))

        predictions = np.where(h > 0.5 ,1 ,0)
        self.predictions = predictions

    def validate(self , y_test) :
        score = np.sum(self.predictions == y_test ) / len(self.predictions)

        print(score)
