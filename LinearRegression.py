import numpy as np
import random
import math

class LinearRegression:
    def __init__(self, learning_rate: float, early_stop: float, num_features: int, sample_size: int): 
        """
        LinearRegression(learning rate, early stop)
            init weights to random numbers
            bias to random numbers
        """
        self.learning_rate = learning_rate
        self.early_stop = early_stop

        # self.w = np.array([[3.5,3.5,5.5]])
        self.w = np.array([[0] * num_features]) # w is horizontal
        # self.b = 1.5
        self.b = 0 #random.randint(1,10)

        # self.rand_h_vector(self.w)
        # self.rand_v_vector(self.b)
        # print(f"init: rand w: {self.w}, rand b: {self.b} ")
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
        curr_error = self.validate(x_val, y_val)
        prev_error = curr_error + 1
        # print(f"train: prev: {prev_error}, curr: {curr_error}, curr-prev: {curr_error-prev_error}")
        # print(f"train: w: {self.w}, b: {self.b}")
        count = 0
        loss = [curr_error]
        # while (curr_error - prev_error) < 0:
        # while abs((curr_error - prev_error)) > self.early_stop:
        while (curr_error - prev_error) < 0 and count <= self.early_stop: 
            yhat = self.predict(x_train) # TODO why are we calling predict?
            # print(f"train: yhat: {yhat}")
            self.w, self.b = self.optimizeSGD(x_train, y_train) 
            # print(f"train: new w: {self.w}, new b: {self.b}")
            prev_error = curr_error
            curr_error = self.validate(x_val, y_val)
            loss.append(curr_error)
            print(f"train: new prev: {prev_error}, new curr: {curr_error}, new curr-prev: {curr_error - prev_error}")
            # user = input("pause")
            count += 1
            if math.isnan(curr_error - prev_error):
                return None
        # print(f"train: FINAL MODEL: w: {self.w}, b: {self.b}")
        print(f"final loss: {curr_error}, num runs: {count}")
        return loss

    def fit(self, x_train, y_train):
        # this is for bias_variance
        # i know this is probably a bit wonky, but whatever
        return self.train(x_train, y_train, x_train, y_train)
    

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        predict(x) -> y^
        y-hat=W*(matrix mult)X + B
        w = [[4,5,6]] (1, feat)
        x = [[1,1,1],[1,1,1]] (samp, feat)
        """
        xT = x.transpose()
        if self.w.shape[1] != xT.shape[0]: # 1xn == nxm
            print(f"predict: wrong size!, w: {self.w.shape}, xT: {xT.shape}, second of w should be same as first of xT")
            return None
        # print(f"predict: w shape: {self.w.shape}, xT shape: {xT.shape}, b: {self.b}")
        yhat = np.array(np.dot(self.w, xT)) + self.b
        # print(f"predict: yhat shape: {yhat.shape}")
        return yhat.transpose()
        

    def lossMSE(self, yhat: np.ndarray, y: np.ndarray) -> float:
        """
        lossMSE(y^, y) -> int (error)
            L = 1/n sum(i=1, n)([Y-hati - Yi]^2)
        """
        # print(f"lossmSE: yhat shape: {yhat.shape}, y shape: {y.shape}, should be equal")
        # print(f"lossMSE: yhat: {yhat}, y: {y}")
        if yhat.shape != y.shape:
            print(f"lossMSE: wrong sizes! yhat: {yhat}, y: {y}")
            return
        n = yhat.shape[0]
        loss = np.sum(((y-yhat)**2))/n
        # print(f"lossMSE: loss: {loss}")
        return loss

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

        # print(f"optimizeGSD: x: {x}, y: {y}, x shape: {x.shape}, y shape: {y.shape}")
        if x.shape[0] != y.shape[0]: # same rows in x as len(y)
            print(f"optimizeSGD: wrong size! both need same rows x: {x.shape}, y: {y.shape}")
            return

        n = y.shape[0]
        # print(f"optimizeSGD: n: {n}")

        # print(f"optimizeSGD: compute yhat: w: {self.w}, xT {x.transpose()}, b: {self.b}")
        yhat = np.array(np.dot(self.w, x.transpose())).transpose() + self.b # result is (samp, 1)
        # print(f"optimizeSGD: compute error: yhat: {yhat}, y: {y}")
        error = (y-yhat)
        # print(f"optimizeSGD: compute dldw: error: {error}, xT: {x.transpose()}, n: {n}")
        dldw = np.array(np.array(np.dot(x.transpose(), error)) * -2/n).transpose() # result is (1, feat)
        # print(f"optimizeSGD: dldw: {dldw}")

        dldb = np.sum(y-yhat) * -2/n
        # print(f"optimizeSGD: dldb: {dldb}")
        # return (self.w - dldw/1000), (self.b - dldb/1000)
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
        # print(f"validate: yhat: {yhat}, y: {y}")
        return self.lossMSE(yhat, y) # determine loss

    def test(self, test_x: np.ndarray):
        """
        test(x-test, y-test)   
            yhat = predict(x)
            computeR2(y, yhat)
        """
        yhat = self.predict(test_x)
        # print(f"R2: {self.R2(yhat, test_y)}")
        # return self.R2(yhat, test_y)
        return yhat

    def R2(self, yhat: np.ndarray, y: np.ndarray):
        """
        computeR2(x, y) -> float 
        """
        """
        https://online.stat.psu.edu/stat462/node/95/
        SSR = regression sum of squares =  sum((yhat-ybar)^2)
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
        # print(f"R-squared: yhat-shape: {yhat.shape}, y shape: {y.shape}, should be same")
        if yhat.shape[0] != y.shape[0]:
            print(f"R-squared: wrong shapes! should be same number of features, yhat: {yhat.shape}, y: {y.shape}")
            return
        ybar = np.average(y)

        # ssr = np.sum((yhat-ybar)**2) # regression sum of squares
        sse = np.sum((y-yhat)**2) # error sum of squares
        ssto = np.sum((y-ybar)**2) # total sum of squares
        # print(f"R-squared: sse: {sse}, ssr: {ssr}, ssto: {ssto}, test ssto: {np.sum((y-ybar)**2)}")
        return 1 - sse/ssto # r^2
"""
(yhat-ybar)**2 + (y-yhat)**2 = (y-ybar)**2
yhat**2-ybaryhar+ybar**2 + y**2-yyhat+yhat**2 = y**2-yybar+ybar**2

2yhat**2 - ybaryhat - yyhat + ybar**2 + y**2 = y**2 - yybar + ybar**2

2yhat**2 - ybaryhat - yyhat + yybar = 0
hm?
"""

def test_rand_1d_array():
    
    x = [0,0,0]
    lr = LinearRegression(0,0,0,0)
    lr.rand_1d_array(x)
    # print(f"x start [0,0,0], x now: {x}")
    return
    
def test_rand_2d_array():
    x = [[0,0,0], [0,0,0], [0,0,0]]
    lr = LinearRegression(0,0,0,0)
    lr.rand_2d_array(x)
    # print(f"x start [[0]*3, [0]*3, [0]*3], x now: {x}")
    return

if __name__ == "__main__":
    # learning_rate = 0.0001
    learning_rate = 0.0001
    early_stop = 0.0001
    num_features = 3
    sample_size = 4
    lr = LinearRegression(learning_rate, early_stop, num_features, sample_size)

    """
    w1 = 3, w2 = 4, w3 = 5, b = 2
    123 - 28, 456 - 64, 789 - 100, 101112 - 136
    131415 - 172 161718 - 208, 192021 - 244, 222324 - 280 
    """

    x_train = np.array([[1,2,3],[4,5,6],[10,11,12], [16,17,18]])
    y_train = np.array([[28],[64],[136],[208]])
    x_val = np.array([[13,14,15],[7,8,9]])
    y_val = np.array([[172],[100]])
    x_test = np.array([[19,20,21],[22,23,24]])
    y_test = np.array([[244], [280]])

    num_runs = lr.train(x_train, y_train, x_val, y_val)
    print(f"num_runs: {num_runs}")
    R2 = lr.test(x_test, y_test)
    print(f"R2: {R2}")




