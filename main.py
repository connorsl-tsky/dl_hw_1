import pandas as pd
from Preprocessing import preprocess, train_val_split, train_test_val_split, separate_labels
from LinearRegression import LinearRegression

Y_HEADER = "TARGET_deathRate"

def main():
    train = pd.read_csv("train.csv")
    train = preprocess(train)
    x_train, y_train = separate_labels(train, Y_HEADER)
    x_train, y_train, x_val, y_val = train_val_split(x_train, y_train, .2)

    """
    early stop 0.9
    1e-13 1673, 1581, 1504, 1519
    1e-12 252, 384, 325, 248, 238, 414, 235
        XXX all of these lead to infinite values
        1e-11 669, 543, 524, 510, 519
        1e-10 106, 110, 106, 122 109
        1e-9 67, 63, 63, 63
        1e-8 45, 45, 45
        1e-7 35, 35, 37
        1e-6 29, 29, 29
        1e-5 25, 25, 25
        1e-4 22, 22, 22, 

    early stop 0.1 
    1e-13 - 3218, 3664, 2466, 2229, 2918
    1e-12 - 4290, 4311, 4814, 5008, 3838

    es 0.01
    1e-13 - 49299
    1e-12 - 8944 i think 1e-12 is generally better
    """

    learning_rate = 1e-13
    early_stop = 0.1
    num_features = x_train.shape[1]
    sample_size = x_train.shape[0]
    lr = LinearRegression(learning_rate, early_stop, num_features, sample_size)
    num_runs = lr.train(x_train, y_train, x_val, y_val)
    print(f"MAIN DONE: num_runs: {num_runs}")

def main_test():
    train = pd.read_csv("train.csv")
    train = preprocess(train)
    x, y = separate_labels(train, Y_HEADER)
    x_train, y_train, x_test, y_test, x_val, y_val = train_test_val_split(x, y, .2)


    learning_rate = 1e-13
    early_stop = 0.001
    num_features = x_train.shape[1]
    sample_size = x_train.shape[0]
    lr = LinearRegression(learning_rate, early_stop, num_features, sample_size)
    num_runs = lr.train(x_train, y_train, x_val, y_val)
    print(f"MAIN TEST DONE: num_runs: {num_runs}")
    R2 = lr.test(x_test, y_test)
    print(f"MAIN TEST R-Squared: {R2}")


if __name__ == "__main__":
    main_test()