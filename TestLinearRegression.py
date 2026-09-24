import LinearRegression as lr
import numpy as np

class TestLinearRegression:
    def __init__(self):
        self.lr = lr.LinearRegression(0,0,2,3)
        self.x = np.matrix([[1,2],[4,5],[7,8]])
        self.lr.w = np.array([4,5])
        self.lr.b = np.array([1,2,3])

    def test_predict(self):
        yhat = self.lr.predict(self.x)
        print(f"yhat predicted: [15, 43, 71], yhat: {yhat}")

    def test_lossMSE(self):
        yhat = np.array([15,43,71])
        y=np.array([12,50,67])
        l = self.lr.lossMSE(yhat,y)
        """
        9 + 49 + 16 = 74/3 = 24.666
        """
        print(f"l predicted: 24.666, l: {l}")

    def test_optimizeSGD(self):
        x = np.matrix([[1,2],[4,5],[7,8]])
        y = np.array[1,2,3]
        self.lr.w = np.array([4,5])
        self.lr.b = np.array([1,2,3])
        w, b = self.lr.optimizeSGD(x, y)
        """
        15
        draw it out dude
        """

if __name__ == "__main__":
    tlr = TestLinearRegression()
    # tlr.test_predict()
    tlr.test_lossMSE()