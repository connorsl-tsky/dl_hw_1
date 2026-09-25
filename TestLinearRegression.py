import LinearRegression as lr
import numpy as np

class TestLinearRegression:
    def __init__(self, ler, es, nf, ss):
        self.lr = lr.LinearRegression(ler, es, nf, ss)
        self.x = np.array([[1,2],[4,5],[7,8]])
        self.lr.w = np.array([[4,5]])
        self.lr.b = 3

    def test_rand_vector(self):
        print("TEST RAND VECTOR")
        x = np.array([[0]*10])
        self.lr.rand_h_vector(x)
        print(f"x: {x}, want: random array [[1,2,3,4,...]]")
        x = np.array([[0]]*10)
        self.lr.rand_v_vector(x)
        print(f"x: {x}, want: random array [[1],[2],[3],...]")
        print()
        return
                
    def test_predict(self):
        print("TEST PREDICT")
        self.lr.w = np.array([[3,4]])
        self.x = np.array([[1,2],[3,4],[5,6]])
        self.lr.b = 3
        yhat = self.lr.predict(self.x)
        print(f"test predict: yhat predicted: [14, 28, 42], yhat: {yhat}")
        print(f"test predict: w: {self.lr.w}, x: {self.x}, b: {self.lr.b}")
        print()
        return

    def test_lossMSE(self):
        print("TEST LOSS MSE")
        yhat = np.array([[15],[43],[71]])
        y = np.array([[12],[50],[67]])
        l = self.lr.lossMSE(yhat,y)
        """
        9 + 49 + 16 = 74/3 = 24.666
        """
        print(f"l predicted: 24.666, l: {l}")
        print()
        return

    def test_optimizeSGD(self):
        print("TEST OPTIMIZE SGD")
        x = np.array([[1,2],[3,4],[5,6]])
        y = np.array([[1],[2],[3]])
        self.lr.w = np.array([[3,4]])
        self.lr.b = 3
        w, b = self.lr.optimizeSGD(x, y)
        print(f"w: want: [[-92.3?, -117.3?]] got: {w}, b: want: -75? got: {b}")
        print()
        return

    def test_r2(self):
        print("TEST R2")
        yhat = np.array([[1,2,3]])
        y = np.array([[2,3,4]])
        r2 = self.lr.R2(yhat, y)
        print(f"r2 want: 0.5 have: {r2}")
        print()
        return

if __name__ == "__main__":
    learning_rate = 0.5
    early_stop = 1
    num_features = 2
    sample_size = 3
    tlr = TestLinearRegression(learning_rate,early_stop,num_features,sample_size)
    tlr.test_predict()
    tlr.test_rand_vector()
    tlr.test_lossMSE()
    tlr.test_optimizeSGD()
    tlr.test_r2()