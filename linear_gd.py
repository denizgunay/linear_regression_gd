import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


class LinearRegressionGD:
    def __init__(self):
        """
        The LinearRegressionGD class constructor.
        """

        # initialize learning rate lr and number of iteration iters
        self.lr = None
        self.iters = None
        # initialize the weights matrix
        self.weights = None
        # bins specifies how many iterations an MSE value will be saved to mse_history.
        self.bins = None
        # mse_history records MSE values in bins intervals.
        self.mse_history = []
        # keeps how many independent variables there are.
        self.n_features = "You should use fit() function first!"
        # keeps the MSE value of the optimal model.
        self.mse = "You should use performance() function first!"
        # keeps the RMSE value of the optimal model.
        self.rmse = "You should use performance() function first!"
        # keeps the MAE value of the optimal model.
        self.mae = "You should use performance() function first!"
        # keeps the R-squared value of the optimal model.
        self.r2 = "You should use performance() function first!"
        # keeps the adjusted R-squared value of the optimal model.
        self.ar2 = "You should use performance() function first!"
        # keeps the SSE value of the optimal model.
        self.sse = "You should use performance() function first!"
        # keeps the SSR value of the optimal model.
        self.ssr = "You should use performance() function first!"
        # keeps the SST value of the optimal model.
        self.sst = "You should use performance() function first!"

    def performance(self, y_predicted, y, verbose=True):
        """
        This function calculates performance metrics such as
        RMSE, MSE, MAE, SSR, SSE, SST, R-squared and Adj. R-squared.

        Args:
            y_predicted (numpy.ndarray): predicted y values
            y (numpy.ndarray): true y values
            verbose (bool, optional): prints performance metrics. Defaults to True.
        """
        self.mse = np.mean(np.sum((y_predicted - y) ** 2))
        self.rmse = np.sqrt(self.mse)
        self.mae = np.mean(np.abs(y - y_predicted))
        self.ssr = np.sum((y_predicted - np.mean(y)) ** 2)
        self.sst = np.sum((y - np.mean(y)) ** 2)
        self.sse = np.sum((y - y_predicted) ** 2)
        self.r2 = 1 - self.sse / self.sst
        self.ar2 = 1 - (((1 - self.r2) * (len(y) - 1)) / (len(y) - self.n_features - 1))
        if verbose:
            print(f"RMSE = {self.rmse}")
            print(f"MSE = {self.mse}")
            print(f"MAE = {self.mae}")
            print(f"SSE = {self.sse}")
            print(f"SSR = {self.ssr}")
            print(f"SST = {self.sst}")
            print(f"R-squared = {self.r2}")
            print(f"Adjusted R-squared = {self.ar2}")

    def predict(self, X):
        """
        This function takes one argument which is a numpy.array of predictor values,
        and returns predicted y values.

        Note: You should use fit() function at least once before using predict() function,
        since the prediction is made with the optimal weights obtained by the fit() function.

        Args:
            X (numpy.ndarray): predictors(input)

        Returns:
            numpy.ndarray: predicted y values
        """
        self.mse = "You should use performance() function first!"
        self.rmse = "You should use performance() function first!"
        self.mae = "You should use performance() function first!"
        self.r2 = "You should use performance() function first!"
        self.ar2 = "You should use performance() function first!"
        self.sse = "You should use performance() function first!"
        self.ssr = "You should use performance() function first!"
        self.sst = "You should use performance() function first!"
        # modify the features X by adding one column with value equal to 1
        ones = np.ones(len(X))
        features = np.c_[ones, X]
        # predict by multiplying the feature matrix with the weight matrix
        y_predicted = np.dot(features, self.weights.T)
        return y_predicted

    def fit(
        self,
        X,
        y,
        init_weights: list = None,
        lr=0.00001,
        iters=1000,
        bins=100,
        verbose=False,
    ):
        """
        This function calculates optimal weights using X(predictors) and Y(true results).

        Args:
            X (numpy.ndarray): predictors
            y (numpy.ndarray): true results
            init_weights (list, optional): initial weights(including bias). Defaults to None.
            lr (float, optional): learning rate. Defaults to 0.00001.
            iters (int, optional): number of iterations. Defaults to 1000.
            bins (int, optional): specifies how many iterations an MSE value will be saved to mse_history. Defaults to 100.
            verbose (bool, optional): prints weights and MSE value in the current iteration. Defaults to False.

        Returns:
            numpy.ndarray: optimal weights(including bias)
        """
        n_samples = len(X)
        ones = np.ones(len(X))
        # modify x, add 1 column with value 1
        features = np.c_[ones, X]
        # initialize the weights matrix
        if init_weights != None:
            if len(init_weights) != features.shape[1]:
                print(f"The length of 'init_weights' should be {features.shape[1]}")
                return
            else:
                self.weights = np.array(init_weights).reshape((1, len(init_weights)))
        else:
            self.weights = np.zeros((1, features.shape[1]))
        self.lr = lr
        self.iters = iters
        self.n_features = X.shape[1]
        self.mse_history = []
        self.bins = bins

        for i in range(self.iters):
            # predicted labels
            y_predicted = np.dot(features, self.weights.T)
            # calculate the error
            error = y_predicted - y
            # compute the partial derivated of the cost function
            dw = (2 / n_samples) * np.dot(features.T, error)
            dw = np.sum(dw.T, axis=0).reshape(1, -1)
            # update the weights matrix
            self.weights -= self.lr * dw

            if i % self.bins == 0:
                self.mse_history.append(np.mean(np.sum(error**2)))
                if verbose:
                    print(
                        f"After {i} iterations: weights = {self.weights}, MSE = {np.mean(np.sum(error**2)):.6f}"
                    )

        if verbose:
            print(
                f"After {self.iters} iterations: weights = {self.weights}, MSE = {np.mean(np.sum(error**2)):.6f} "
            )
        return self.weights

    def visualize(self, size=(15, 6), bottom=0, top=None, left=-10, right=None):
        """
        This function plots the cost and iteration graph.

        Args:
            size (tuple, optional): (width of plot, height of plot). Defaults to (15, 6).
            bottom (int or float, optional): lowest value of y axis. Defaults to 0.
            top (int or float, optional): highest value of y axis. Defaults to None.
            left (int, optional): lowest value of x axis. Defaults to -10.
            right (int, optional): highest value of x axis. Defaults to None.
        """
        if top == None:
            top = max(self.mse_history)
        if right == None:
            right = (self.iters // self.bins) * self.bins
        plt.figure(figsize=size)
        plt.title("Cost and Iteration", fontsize=20)
        plt.xlabel("Iterations")
        plt.ylabel("MSE")
        plt.plot(range(0, self.iters, self.bins), self.mse_history, color="b")
        plt.ylim(bottom=bottom, top=top)
        plt.xlim(left=left, right=right)
        plt.show(block=True)

    def cross_validate(
        self,
        X,
        y,
        lr=0.00001,
        iters=1000,
        k=10,
        scoring="r2",
        init_weights: list = None,
        verbose=True,
    ):
        """
        This function applies K-fold cross validation to the dataset and assess the performance
        of the model in a robust and reliable manner.

        Args:
            X (numpy.ndarray): predictors(input)
            y (numpy.ndarray): true results
            lr (float, optional): learning rate. Defaults to 0.00001.
            iters (int, optional): number of iterations. Defaults to 1000.
            k (int, optional): number of folds which the original dataset is divided. Defaults to 10.
            scoring (str, optional): the performance metric to be calculated. Defaults to "r2".
            init_weights (list, optional): initial weights(including bias). Defaults to None.
            verbose (bool, optional): prints the average score and score list. Defaults to True.

        Returns:
            tuple: (average score, score list)
        """
        if scoring not in ["r2", "ar2", "mse", "rmse", "mae"]:
            print(
                "The 'scoring' parameter is invalid. Available ones are the following:"
            )
            print(["r2", "ar2", "mse", "rmse", "mae"])
            return
        scores = []
        kf = KFold(n_splits=k, shuffle=True)
        for train_index, test_index in kf.split(X):
            model = LinearRegressionGD()
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(x_train, y_train, lr=lr, iters=iters, init_weights=init_weights)
            y_pred = model.predict(x_test)
            model.performance(y_pred, y_test, verbose=False)
            if scoring == "r2":
                scores.append(model.r2)
            elif scoring == "ar2":
                scores.append(model.ar2)
            elif scoring == "mse":
                scores.append(model.mse)
            elif scoring == "rmse":
                scores.append(model.rmse)
            elif scoring == "mae":
                scores.append(model.mae)
        if verbose:
            print(f"{scoring} scores : {scores}")
            print(f"Average {scoring} : {np.mean(scores)}")
        return np.mean(scores), scores
