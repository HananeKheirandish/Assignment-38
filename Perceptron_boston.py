import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

class Perceptron:
    def __init__(self, learning_rate=0.05, epochs=2):
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X_train, Y_train):
        self.X_train = np.array(X_train)
        self.Y_train = np.array(Y_train)
        
        N = X_train.shape[0]
        
        # init weights
        self.w = np.random.rand(2, 1)

        fig = plt.figure(figsize=(12, 6))

        Errors = []

        x_range = np.arange(self.X_train[:,0].min(), self.X_train[:,0].max())
        y_range = np.arange(self.X_train[:,1].min(), self.X_train[:,1].max())
        x, y = np.meshgrid(x_range, y_range)
        
        # Train
        for epoch in range(self.epochs):
            for i in range(N):
                x_train = X_train[i, :]
                y_pred = np.matmul(x_train, self.w)
                e = Y_train[i] - y_pred

                # update weights
                x_train = x_train.reshape(-1, 1)
                self.w += e * self.learning_rate * x_train

                # visualization
                fig.clear()
                Y_pred = np.matmul(self.X_train, self.w)
                
                ax1 = fig.add_subplot(121, projection= '3d')
                ax1.clear()
                ax1.scatter(self.X_train[:, 0], self.X_train[:, 1], self.Y_train, c='#0000ff')
                z = self.w[0] * x + self.w[1] * y
                ax1.plot_surface(x, y, z, alpha=0.5)
                ax1.set_xlabel('CRIM')
                ax1.set_ylabel("RM")
                ax1.set_zlabel("MEDV")

                Error = np.mean(np.abs(self.Y_train - Y_pred))
                Errors.append(Error)
                
                ax2 = fig.add_subplot(122)
                ax2.plot(Errors)

                plt.pause(0.01)
        plt.show()
        
    def predict(self, X_test):
        Y_pred = np.matmul(X_test, self.w)
        return Y_pred
    
    def evaluate(self, X_test, Y_test):
        Y_pred = np.matmul(X_test, self.w)
        Error = np.abs(Y_test - Y_pred)
        MSE = np.mean(Error**2)
        return MSE

data = load_boston()

boston = pd.DataFrame(data.data, columns=data.feature_names)
boston['MEDV'] = data.target
boston.head()

X = boston[['CRIM', 'RM']].to_numpy()
Y = boston[['MEDV']].to_numpy()
Y = Y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y)

perceptron = Perceptron()
perceptron.fit(X, Y)