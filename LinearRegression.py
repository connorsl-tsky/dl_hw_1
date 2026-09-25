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

        self.w = np.array([[0] * num_features]) # w is horizontal
        self.b = random.randint(1,10)

        self.rand_h_vector(self.w)
        # self.rand_v_vector(self.b)
        print(f"init: rand w: {self.w}, rand b: {self.b} ")
        return

    def rand_h_vector(self, x: np.ndarray):
        for i in range(len(x[0])):
            x[0][i] = random.randint(1,10)
        return

    def rand_v_vector(self, x: np.ndarray):
        for i in range(len(x)):
            x[i][0] = random.randint(1,10)
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

    def train(self, x_train, y_train, x_val, y_val):
        """
        train(x-train, y-train, x-val, y-val) -> void (updates w/b)
            prev error = curr error ? curr error : 0
            curr error = validate(x-val, y-val)
            while (prev error - curr error) > 0
                yhat = predict(x-train) 
                w, b = optimizerSGD(yhat, y-train)
        """
        prev_error = 0
        curr_error = self.validate(x_val, y_val)
        print(f"train: prev: {prev_error}, curr: {curr_error}, curr-prev: {curr_error-prev_error}")
        print(f"train: w: {self.w}, b: {self.b}")
        while True:#(curr_error - prev_error) > 0:
            yhat = self.predict(x_train) # TODO why are we calling predict?
            print(f"train: yhat: {yhat}")
            self.w, self.b = self.optimizeSGD(x_train, y_train) 
            print(f"train: new w: {self.w}, new b: {self.b}")
            prev_error = curr_error
            curr_error = self.validate(x_val, y_val)
            print(f"train: new prev: {prev_error}, new curr: {curr_error}, new prev-curr: {prev_error-curr_error}")
            user = input("pause")
        return

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        predict(x) -> y^
        y-hat=W*(matrix mult)X + B
        w = [[4,5,6]] (1, feat)
        x = [[1,1,1],[1,1,1]] (samp, feat)
        """
        xT = x.transpose()
        if self.w.shape[1] != xT.shape[0]: # 1xn == nxm
            print(f"predict: wrong size!, w: {len(self.w)}, xT: {xT.shape}")
            return None
        print(f"predict: w shape: {self.w.shape}, xT shape: {xT.shape}, b: {self.b}")
        yhat = np.array(np.dot(self.w, xT)) + self.b
        print(f"predict: yhat shape: {yhat.shape}")
        return yhat.transpose()
        

    def lossMSE(self, yhat: np.ndarray, y: np.ndarray) -> float:
        """
        lossMSE(y^, y) -> int (error)
            L = 1/n sum(i=1, n)([Y-hati - Yi]^2)
        """
        if len(yhat) != len(y):
            print(f"lossMSE: wrong sizes! yhat: {yhat}, y: {y}")
        total = 0
        n = len(yhat)
        for i in range(n):
            total += (yhat[i]-y[i])**2
        return total/n

    def optimizeSGD(self, x: np.ndarray, y: np.ndarray):
        """
        optimizerSGD(y^, y) -> w, b
            dL/dw = 2/n sum(i=1, n) x((wx + b) - y)
            dL/db = 2/n sum(i=1, n) (wx + b) - y
            w = w - I * dL/dw
            b = b - I * dL/db
        x is (samp,feat)
        y is (samp, 1)
        w is (1, feat)
        b is (samp, 1)
        """

        if x.shape[0] != y.shape[0]: # same rows in x as len(y)
            print(f"optimizeSGD: wrong size! both need same rows x: {x.shape}, y: {y.shape}")

        n = len(y)

        print(f"optimizeSGD: w: {self.w}, x dim {x.shape}, b: {self.b}")
        yhat = np.array(np.dot(self.w, x.transpose())) + self.b # result is (samp, 1)
        print(f"optimizeSGD: yhat: {yhat}, yhat dim {yhat.shape}, y dim {y.shape}")
        dldw = np.array(np.dot((yhat-y.transpose()), x)) * -2/n # result is (1, feat)
        print(f"optimizeSGD: dldw: {dldw}")

        dldb = np.sum(yhat-y.transpose()) * -2/n
        # print(f"optimizeSGD: dldb: {dldb}")
        return (self.w - self.learning_rate*dldw), (self.b - self.learning_rate*dldb)

    def validate(self, x, y):
        """
        validate(x-validation, y-validation) -> int (error)
            yhat = predict()
            return lossMSE(yhat, y-val)
        x is (samp, feat)
        y is (samp, 1)
        """
        yhat = self.predict(x) # make predictions
        print(f"validate: yhat: {yhat}, y: {y}")
        return self.lossMSE(yhat, y) # determine loss

    def test(self, x, y):
        """
        test(x-test, y)   
            yhat = predict(x)
            computeR2(y, yhat)
        """
        yhat = self.predict(x)
        print(f"R2: {self.R2(yhat, y)}")
        return

    def R2(self, yhat: np.ndarray, y: np.ndarray):
        """
        computeR2(x, y) -> float 
        """
        """
        https://online.stat.psu.edu/stat462/node/95/
        SSR = regression sum of squares =  sum((yhat-y)^2)
        SSE = error sum of squares = sum((y-yhat)^2)
        SSTO = total sum of squares = sum((y-ybar)^2)
        SSTO = SSR + SSE
        R^2 = SSR/SSTO = 1-SSE/SSTO
        https://online.stat.psu.edu/stat462/node/131/ 
        adj r2 = 1-((n-1)/(n-(k+1)))(1-R^2) - but we might not calculate it
        idk
        yhat (samp, 1)
        y (samp, 1)
        """
        if yhat.shape[0] != y.shape[0]:
            print(f"R-squared: wrong shapes! should be same number of features, yhat: {yhat.shape}, y: {y.shape}")
            return

        sse = np.sum((y-yhat)**2) # error sum of squares
        ssr = np.sum((yhat-y)**2) # regression sum of squares
        ssto = ssr + sse # total sum of squares
        return ssr / ssto # r^2


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
    learning_rate = 0.5
    early_stop = 0
    num_features = 3
    sample_size = 4
    lr = LinearRegression(learning_rate, early_stop, num_features, sample_size)

    x_train = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    y_train = np.array([[0],[1],[0],[1]])
    x_val = np.array([[13,14,15],[16,17,18]])
    y_val = np.array([[0],[1]])
    x_test = np.array([[19,20,21],[22,23,24]])

    lr.train(x_train, y_train, x_val, y_val)
    




