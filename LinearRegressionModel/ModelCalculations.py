import numpy as np

class ModelCalculations:
    def __init__(self, w, b, x, y, iteration, alpha, m):
        self.w = w
        self.b = b
        self.x = x
        self.y = y
        self.iteration = iteration
        self.alpha = alpha
        self.m = m

    @staticmethod
    def calculate_y_pre(w, b, x):
        y_pre = w * x + b
        return y_pre

    def calculate_cost_function(self, y_pred):
        cost_function = (1 / (2 * self.m)) * np.sum(np.square(y_pred - self.y))
        return cost_function

    @staticmethod
    def calculate_gradiant(m, y_pred, y, x):
        dw = (1 / m) * np.sum((y_pred - y) * x)
        db = (1 / m) * np.sum(y_pred - y)
        return dw, db

    def calculate_gradiant_descent(self, dw, db):
        self.w -= self.alpha * dw
        self.b -= self.alpha * db

    def linear_regression(self):
        cost = None
        for i in range(self.iteration):

            y_pred = self.calculate_y_pre(w=self.w, b=self.b, x=self.x)

            dw, db = self.calculate_gradiant(m=self.m, y_pred=y_pred, y=self.y, x=self.x)

            if np.isnan(dw) or np.isnan(db) or np.isinf(dw) or np.isinf(db):
                print(f"!!! Error: NaN/Inf detected at iteration {i}. Lower Alpha!")
                break

            self.calculate_gradiant_descent(dw=dw, db=db)

            if i % 100 == 0:
                cost = self.calculate_cost_function(y_pred=y_pred)
                print(f"Iteration {i}: Cost {cost:.6f}")

        return cost