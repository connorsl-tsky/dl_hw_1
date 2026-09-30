import pandas as pd
from Preprocessing import preprocess, train_val_split, train_test_val_split, separate_labels, separate_labels_ids, separate_labels_ids_test
from LinearRegression import LinearRegression
from DNN import DNN
import keras
import numpy as np
import matplotlib.pyplot as plt

Y_HEADER = "TARGET_deathRate"

def main_lr():
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
    train_ids, x, y = separate_labels_ids(train, Y_HEADER)
    x_train, y_train, x_val, y_val = train_val_split(x, y, .2)
    test = pd.read_csv("test.csv")
    test = preprocess(test)
    test_ids, x_test = separate_labels_ids_test(test)


    learning_rate = 1e-13
    # early_stop = 0.001
    early_stop = 0.1
    num_features = x_train.shape[1]
    sample_size = x_train.shape[0]
    lr = LinearRegression(learning_rate, early_stop, num_features, sample_size)
    num_runs = lr.train(x_train, y_train, x_val, y_val)
    print(f"MAIN TEST DONE: num_runs: {num_runs}")
    yhat = lr.test(x_test)
    yhat = np.reshape(yhat, (1, yhat.shape[0]))[0]
    # print(f"MAIN TEST R-Squared: {R2}")
    print(test_ids.to_numpy(), yhat)

def main():
    """
    all 5 models
    trains on train.csv, and outputs predictions and plots to test.csv
    """

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    train = preprocess(train)
    test = preprocess(test)
    train_ids, x_train, y_train = separate_labels_ids(train, Y_HEADER)
    test_ids, x_test = separate_labels_ids_test(test)
    x_train_lr, y_train_lr, x_val_lr, y_val_lr = train_val_split(x_train, y_train, 0.2) 

    # lr_learning_rate = 1e-12
    # lr_learning_rate = 5e-12
    lr_learning_rate = 5e-12
    # lr_early_stop = 0.001
    lr_early_stop = 20
    lr_num_features = x_train.shape[1]
    lr_sample_size = x_train.shape[0]
    lr = LinearRegression(lr_learning_rate, lr_early_stop, lr_num_features, lr_sample_size)
    lr_loss = lr.train(x_train_lr, y_train_lr, x_val_lr, y_val_lr)
    # predictions
    predictions = lr.test(x_test)
    predictions = np.reshape(predictions, (1, predictions.shape[0]))[0]
    test_ids = test_ids.to_numpy()
    # output
    output(test_ids, predictions, "lr_submission.csv")

    # DNN 8 out
    layers = [8]
    # loss = 'mse'
    loss = 'mae'
    learning_rate=1e-4
    metrics = [keras.metrics.R2Score(), keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    epochs = 20
    batch_size = 32
    
    dnn1 = DNN(layers, loss, learning_rate, metrics, epochs, batch_size)
    dnn1.train(x_train, y_train)
    # test

    ids, predictions = dnn1.test(test)
    output(ids, predictions, "dnn1_submission.csv")
    
    # DNN 16 8 out
    layers = [16, 8]
    # loss = 'mse'
    learning_rate=1e-4
    metrics = [keras.metrics.R2Score(), keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    epochs = 20
    batch_size = 32
    
    dnn2 = DNN(layers, loss, learning_rate, metrics, epochs, batch_size)
    dnn2.train(x_train, y_train)

    ids, predictions = dnn2.test(test)
    output(ids, predictions, "dnn2_submission.csv")

    # DNN 16 8 4 out
    layers = [16, 8, 4]
    # loss = 'mse'
    learning_rate=1e-4
    metrics = [keras.metrics.R2Score(), keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    epochs = 20
    batch_size = 32
    
    dnn3 = DNN(layers, loss, learning_rate, metrics, epochs, batch_size)
    dnn3.train(x_train, y_train)

    ids, predictions = dnn3.test(test)
    output(ids, predictions, "dnn3_submission.csv")

    # DNN 30 16 8 4 out
    layers = [30, 16, 8, 4]
    # loss = 'mse'
    learning_rate=1e-5
    metrics = [keras.metrics.R2Score(), keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    epochs = 20
    batch_size = 32
    
    dnn4 = DNN(layers, loss, learning_rate, metrics, epochs, batch_size)
    dnn4.train(x_train, y_train)

    ids, predictions = dnn4.test(test)
    output(ids, predictions, "dnn4_submission.csv")

    dnn1_loss = dnn1.history.history['loss']
    dnn2_loss = dnn2.history.history['loss']
    dnn3_loss = dnn3.history.history['loss']
    dnn4_loss = dnn4.history.history['loss']
    plot(lr_loss, dnn1_loss, dnn2_loss, dnn3_loss, dnn4_loss)
         
def output(ids, labels, filename):
    outdf = pd.DataFrame({"id": ids, "TARGET_deathRate": labels})
    outdf.to_csv(filename, index=False)
    return

def plot(lr_loss, dnn1_loss, dnn2_loss, dnn3_loss, dnn4_loss, dnn5_loss):
    """
    i think these can be arrays of losses
    """
    lr_epochs = range(1, len(lr_loss)+1)
    dnn1_epochs = range(1, len(dnn1_loss)+1)
    dnn2_epochs = range(1, len(dnn2_loss)+1)
    dnn3_epochs = range(1, len(dnn3_loss)+1)
    dnn4_epochs = range(1, len(dnn4_loss)+1)
    dnn5_epochs = range(1, len(dnn5_loss)+1)
    # plt.plot(lr_epochs, lr_loss, label="Linear Regression")
    plt.plot(dnn1_epochs, dnn1_loss, label="DNN1 - 8 - Output")
    plt.plot(dnn2_epochs, dnn2_loss, label="DNN2 - 16 - 8 - Output")
    plt.plot(dnn3_epochs, dnn3_loss, label="DNN3 - 16 - 8 - 4 - Output")
    plt.plot(dnn4_epochs, dnn4_loss, label="DNN4 - 30 - 16 - 8 - 4 - Output")
    plt.plot(dnn5_epochs, dnn5_loss, label="DNN5 - 8 - 8 - Output")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return

def main_test_2():
    """
    all 5 models
    but we only use train.csv
    and we split it into training, testing, and validations sets
    but validation only needed for LR input bc it doesn't do it itself
    the DNNs do it themselves
    """

    data = pd.read_csv("train.csv")
    data = preprocess(data)
    train_ids, x, y = separate_labels_ids(data, Y_HEADER)
    # pretty sure we don't need the ids
    x_train_lr, y_train_lr, x_test, y_test, x_val_lr, y_val_lr = train_test_val_split(x, y, 0.2) 
    x_train = np.append(x_train_lr, x_val_lr, axis=0)
    y_train = np.append(y_train_lr, y_val_lr, axis=0)

    # lr_learning_rate = 1e-12
    # lr_learning_rate = 5e-12
    lr_learning_rate = 5e-12
    # lr_early_stop = 0.001
    lr_early_stop = 40
    lr_num_features = x_train_lr.shape[1]
    lr_sample_size = x_train_lr.shape[0]
    lr = LinearRegression(lr_learning_rate, lr_early_stop, lr_num_features, lr_sample_size)
    lr_loss = lr.train(x_train_lr, y_train_lr, x_val_lr, y_val_lr)
    # predictions
    predictions = lr.test(x_test)
    pred_mean = predictions.mean(axis=0)
    lr_test_bias = ((y_test - pred_mean)**2).mean()
    lr_test_var = predictions.var(axis=0).mean()
    lr_r2 = lr.R2(predictions, y_test)

    predictions = lr.test(x_train)
    pred_mean = predictions.mean(axis=0)
    lr_train_bias = ((y_train - pred_mean)**2).mean()
    lr_train_var = predictions.var(axis=0).mean()
    

    # output

    """
    https://www.geeksforgeeks.org/machine-learning/bias-vs-variance-in-machine-learning/
    preds = np.array(preds)
    y_pred_mean = preds.mean(axis=0)
    bias_sq = ((y_test - y_pred_mean)**2).mean()
    variance = preds.var(axis=0).mean()
    """

    # DNN 8 out
    layers = [8]
    # loss = 'mse'
    loss = 'mae'
    learning_rate=1e-4
    metrics = [keras.metrics.R2Score(), keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    epochs = 20
    batch_size = 32
    
    dnn1 = DNN(layers, loss, learning_rate, metrics, epochs, batch_size)
    dnn1.train(x_train, y_train)

    predictions = np.array(dnn1.model.predict(x_test))
    pred_mean = predictions.mean(axis=0)
    dnn1_test_bias = ((y_test - pred_mean)**2).mean()
    dnn1_test_var = predictions.var(axis=0).mean()

    predictions = np.array(dnn1.model.predict(x_train))
    pred_mean = predictions.mean(axis=0)
    dnn1_train_bias = ((y_train - pred_mean)**2).mean()
    dnn1_train_var = predictions.var(axis=0).mean()

    dnn1_l, dnn1_r2, dnn1_mse, dnn1_mae = dnn1.model.evaluate(x_test, y_test)

    
    # DNN 16 8 out
    layers = [16, 8]
    # loss = 'mse'
    learning_rate=1e-4
    metrics = [keras.metrics.R2Score(), keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    epochs = 20
    batch_size = 32
    
    dnn2 = DNN(layers, loss, learning_rate, metrics, epochs, batch_size)
    dnn2.train(x_train, y_train)

    predictions = np.array(dnn2.model.predict(x_test))
    pred_mean = predictions.mean(axis=0)
    dnn2_test_bias = ((y_test - pred_mean)**2).mean()
    dnn2_test_var = predictions.var(axis=0).mean()

    predictions = np.array(dnn2.model.predict(x_train))
    pred_mean = predictions.mean(axis=0)
    dnn2_train_bias = ((y_train - pred_mean)**2).mean()
    dnn2_train_var = predictions.var(axis=0).mean()

    dnn2_l, dnn2_r2, dnn2_mse, dnn2_mae = dnn2.model.evaluate(x_test, y_test)


    # DNN 16 8 4 out
    layers = [16, 8, 4]
    # loss = 'mse'
    learning_rate=1e-4
    metrics = [keras.metrics.R2Score(), keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    epochs = 20
    batch_size = 32
    
    dnn3 = DNN(layers, loss, learning_rate, metrics, epochs, batch_size)
    dnn3.train(x_train, y_train)

    predictions = np.array(dnn3.model.predict(x_test))
    pred_mean = predictions.mean(axis=0)
    dnn3_test_bias = ((y_test - pred_mean)**2).mean()
    dnn3_test_var = predictions.var(axis=0).mean()

    predictions = np.array(dnn3.model.predict(x_train))
    pred_mean = predictions.mean(axis=0)
    dnn3_train_bias = ((y_train - pred_mean)**2).mean()
    dnn3_train_var = predictions.var(axis=0).mean()

    dnn3_l, dnn3_r2, dnn3_mse, dnn3_mae = dnn3.model.evaluate(x_test, y_test)


    # DNN 30 16 8 4 out
    layers = [30, 16, 8, 4]
    # loss = 'mse'
    learning_rate=1e-5
    metrics = [keras.metrics.R2Score(), keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    epochs = 20
    batch_size = 32
    
    dnn4 = DNN(layers, loss, learning_rate, metrics, epochs, batch_size)
    dnn4.train(x_train, y_train)

    predictions = np.array(dnn4.model.predict(x_test))
    pred_mean = predictions.mean(axis=0)
    dnn4_test_bias = ((y_test - pred_mean)**2).mean()
    dnn4_test_var = predictions.var(axis=0).mean()

    predictions = np.array(dnn4.model.predict(x_train))
    pred_mean = predictions.mean(axis=0)
    dnn4_train_bias = ((y_train - pred_mean)**2).mean()
    dnn4_train_var = predictions.var(axis=0).mean()

    dnn4_l, dnn4_r2, dnn4_mse, dnn4_mae = dnn4.model.evaluate(x_test, y_test)


    # DNN 8 8 out
    layers = [8, 8]
    # loss = 'mse'
    learning_rate=1e-5
    metrics = [keras.metrics.R2Score(), keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    epochs = 20
    batch_size = 32
    
    dnn5 = DNN(layers, loss, learning_rate, metrics, epochs, batch_size)
    dnn5.train(x_train, y_train)

    predictions = np.array(dnn5.model.predict(x_test))
    pred_mean = predictions.mean(axis=0)
    dnn5_test_bias = ((y_test - pred_mean)**2).mean()
    dnn5_test_var = predictions.var(axis=0).mean()

    predictions = np.array(dnn5.model.predict(x_train))
    pred_mean = predictions.mean(axis=0)
    dnn5_train_bias = ((y_train - pred_mean)**2).mean()
    dnn5_train_var = predictions.var(axis=0).mean()

    dnn5_l, dnn5_r2, dnn5_mse, dnn5_mae = dnn5.model.evaluate(x_test, y_test)


    """
    underfit: high bias = oversimplifies, low variance, bad at testing/training
    overfit: high var = too sensitive to small changes, low bias, bad at testing
    want both to be moderate
    """
    print("\tLR\tDNN1\tDNN2\tDNN3\tDNN4\tDNN5")
    print(f"TRBIAS:\t{lr_train_bias:.2f}\t{dnn1_train_bias:.2f}\t{dnn2_train_bias:.2f}\t{dnn3_train_bias:.2f}\t{dnn4_train_bias:.2f}\t{dnn5_train_bias:.2f}")
    print(f"TRVAR:\t{lr_train_var:.2f}\t{dnn1_train_var:.2f}\t{dnn2_train_var:.2f}\t{dnn3_train_var:.2f}\t{dnn4_train_var:.2f}\t{dnn5_train_var:.2f}")
    print(f"TEBIAS:\t{lr_test_bias:.2f}\t{dnn1_test_bias:.2f}\t{dnn2_test_bias:.2f}\t{dnn3_test_bias:.2f}\t{dnn4_test_bias:.2f}\t{dnn5_test_bias:.2f}")
    print(f"TEVAR:\t{lr_test_var:.2f}\t{dnn1_test_var:.2f}\t{dnn2_test_var:.2f}\t{dnn3_test_var:.2f}\t{dnn4_test_var:.2f}\t{dnn5_test_var:.2f}")
    print()
    print("TEST R2")
    print("LR\tDNN1\tDNN2\tDNN3\tDNN4\tDNN5")
    print(f"{lr_r2:.2f}\t{dnn1_r2:.2f}\t{dnn2_r2:.2f}\t{dnn3_r2:.2f}\t{dnn4_r2:.2f}\t{dnn5_r2:.2f}")
    

    dnn1_loss = dnn1.history.history['loss']
    dnn2_loss = dnn2.history.history['loss']
    dnn3_loss = dnn3.history.history['loss']
    dnn4_loss = dnn4.history.history['loss']
    dnn5_loss = dnn5.history.history['loss']
    plot(lr_loss, dnn1_loss, dnn2_loss, dnn3_loss, dnn4_loss, dnn5_loss)

    return

"""
TDL 9/30/26
- *test functions on the whole test dataset
    - *DNN 9/30/26 1240pm
    - *LR 9/30/26 1258pm
- *output results so it looks like sample_submission.csv
- *graph loss for all models
    https://stackoverflow.com/questions/52614922/how-to-plot-training-loss-and-accuracy-curves-for-a-mlp-model-in-keras
- *turn off print output
- *tune hyperparameters a bit 214pm 
- finish report
    - 234pm
    - *split training into train, test, and var - new main function - 248pm
    - *calculate bias and variance for all models - 308pm
    - *bias and var for training to compare - 315pm shoulda made a function :(
    - *calculate r2 for each model
    - run AI on this stuff
- add my own model and resubmit pictures
- re-answer the data questions and make them better
- AI report 
- best model weights
- l1/l2 regularization?
- tune hyperparameters

not going to add MAE for LR. i'll take the L


"""


if __name__ == "__main__":
    print("hello world")
    main_test_2()