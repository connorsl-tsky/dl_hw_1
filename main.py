import pandas as pd
from Preprocessing import preprocess, train_val_split, separate_labels
from LinearRegression import LinearRegression

Y_HEADER = "TARGET_deathRate"

if __name__ == "__main__":
    train = pd.read_csv("train.csv")
    train = preprocess(train)
    x_train, y_train = separate_labels(train, Y_HEADER)
    x_train, y_train, x_val, y_val = train_val_split(x_train, y_train, .2)

    """
    0.0000000000001
    early stop 0.9
    1e-13 1673, 1581, 1504, 1519
    1e-12 252, 384, 325, 248, 238, 414, 235
    1e-11 669, 543, 524, 510, 519
    1e-10 106, 110, 106, 122
    """

    learning_rate = 1e-10
    early_stop = 0.9
    num_features = x_train.shape[1]
    sample_size = x_train.shape[0]
    lr = LinearRegression(learning_rate, early_stop, num_features, sample_size)
    num_runs = lr.train(x_train, y_train, x_val, y_val)
    print(f"DONE: num_runs: {num_runs}")
