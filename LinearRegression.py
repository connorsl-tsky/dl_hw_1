import numpy as np
import random

class LinearRegression:
    def __init__(self, learning_rate: float, early_stop: float, num_features: int, sample_size: int): 
        """
        LinearRegression(learning rate, early stop)
            init weights to random numbers
            bias to random numbers
        """
        self.learning_rate = learning_rate
        self.early_stop = early_stop

        self.w = np.array([0] * num_features)
        self.b = np.array([0] * sample_size)

        self.rand_1d_array(self.w)
        self.rand_1d_array(self.b)
        return

    def rand_1d_array(self, x: np.ndarray[float]):
        for i in range(len(x)):
            x[i] = random.randint(1,10)
        return

    def rand_2d_array(self, x: list[list[float]]):
        for i in range(len(x)):
            for j in range(len(x[i])):
                x[i][j] = random.randint(1,10)
        return

    def train(self):
        pass

    def predict(self, x: np.matrix) -> np.ndarray:
        """
        predict(x) -> y^
        y-hat=W*(matrix mult)X + B
        w = [4,5,6]
        x = [[1,1,1],[1,1,1]]
        """
        print("PREDICT")
        xT = x.transpose()
        if len(self.w) != xT.shape[0]: # 1xn == nxm
            print(f"predict: wrong size!, w: {len(self.w)}, xT: {xT.shape}")
            return None
        return np.dot(self.w, xT) + self.b
        

    def lossMSE(self, yhat: np.ndarray, y: np.ndarray) -> float:
        """
        lossMSE(y^, y) -> int (error)
            L = 1/n sum(i=1, n)([Y-hati - Yi]^2)
        """
        if len(yhat) != len(y):
            print(f"lossMSE: wrong sizes! yhat: {len(yhat)}, y: {y}")
        total = 0
        n = len(yhat)
        for i in range(n):
            total += (yhat[i]-y[i])**2
        return total/n

    def optimizeSGD(self, x: np.matrix, y: np.ndarray):
        """
        optimizerSGD(y^, y) -> w, b
            dL/dw = 2/n sum(i=1, n) x((wx + b) - y)
            dL/db = 2/n sum(i=1, n) (wx + b) - y
            w = w - I * dL/dw
            b = b - I * dL/db
        """
        if x.shape[0] != len(y): # same rows in x as len(y)
            print(f"optimizeSGD: wrong size! x: {x.shape}, x rows: {x.shape[0]}, y: {len(y)}")

        n = len(y)

        dLdW = 0
        for i in range(n):
            yhat = np.dot(x[i], self.w) + self.b[i]
            dLdW += (yhat-y) * x[i]
        dLdW *= 2/n

        dLdB = 0
        for i in range(n):
            yhat = np.dot(x[i], self.w) + self.b[i]
            dLdW += (yhat-y)
        dLdB *= 2/n
        return (self.w - self.learning_rate*dLdW), (self.b - self.learning_rate*dLdB)

    def validate(self):
        pass

    def test(self):
        pass

    def R2(self):
        pass


def test_rand_1d_array():
    x = [0,0,0]
    lr = LinearRegression(0,0,0,0)
    lr.rand_1d_array(x)
    print(f"x start [0,0,0], x now: {x}")
    return
    
def test_rand_2d_array():
    x = [[0,0,0], [0,0,0], [0,0,0]]
    lr = LinearRegression(0,0,0,0)
    lr.rand_2d_array(x)
    print(f"x start [[0]*3, [0]*3, [0]*3], x now: {x}")
    return

if __name__ == "__main__":
    x_train = np.matrix([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    y_train = np.array([0,1,0,1])
    x_val = np.matrix([[13,14,15],[16,17,18]])
    y_val = np.array([0,1])
    x_test = np.matrix([[19,20,21],[22,23,24]])
    test_rand_1d_array()
    test_rand_2d_array()




"""
train(x-train, y-train, x-val, y-val) -> void (updates w/b)
    prev error = curr error ? curr error : 0
    curr error = validate(x-val, y-val)
    while (prev error - curr error) > 0
        yhat = predict(x-train) 
        w, b = optimizerSGD(yhat, y-train)

validate(x-validation, y-validation) -> int (error)
    yhat = predict()
    return lossMSE(yhat, y-val)

test(x-test)   
    yhat = predict(x)
    computeR2(x, yhat)

computeR2(x, y) -> float 
    n = len(x) = len(y)
    compute sumx, sumy, sum(x^2), sum(y^2), sum(xy)
    return n*Sum(xy)-(SumX)(SumY) / sqrt([nSum(X^2)-(Sumx)^2][nSum(y^2)-(SumY)^2])
"""