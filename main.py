import pandas as pd
from Preprocessing import preprocess, train_val_split, train_test_val_split, separate_labels, separate_labels_ids, separate_labels_ids_test, scale_inputs, scale_labels
from LinearRegression import LinearRegression
from DNN import DNN
import keras
import numpy as np
import matplotlib.pyplot as plt

Y_HEADER = "TARGET_deathRate"

def main():
    """
    just dnn1 - my best model
    trains on train.csv, and outputs predictions and plots to test.csv
    """

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    train = preprocess(train)
    test = preprocess(test)
    train = train.reindex(np.random.permutation(train.index)) # shuffle
    test = test.reindex(np.random.permutation(test.index)) # shuffle
    train_ids, x_train, y_train = separate_labels_ids(train, Y_HEADER)
    x_train = scale_inputs(x_train)
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train = scale_labels(y_train)
    test_ids, x_test = separate_labels_ids_test(test)
    x_test = scale_inputs(x_test)

    learning_rate = 1e-2

    # DNN 8 out
    layers = [8]
    loss = 'mse'
    metrics = [keras.metrics.R2Score(), keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    epochs = 200
    batch_size = 32
    
    dnn1 = DNN(layers, loss, learning_rate, metrics, epochs, batch_size)
    dnn1.train(x_train, y_train)
    # test

    ids, predictions = dnn1.test(test)
    predictions = dnn1.model.predict(x_test)
    predictions = np.reshape(predictions, (1, predictions.shape[0]))[0]
    predictions = (predictions * y_std) + y_mean # unscale predictions
    output(ids, predictions, "dnn1_submission.csv")
    dnn1.model.save('weights.keras')
    
    return

def output(ids, labels, filename):
    outdf = pd.DataFrame({"id": ids, "TARGET_deathRate": labels})
    outdf.to_csv(filename, index=False)
    return

def plot(lr_loss, dnn1_loss, dnn2_loss, dnn3_loss, dnn4_loss, dnn5_loss):
    lr_epochs = range(1, len(lr_loss)+1)
    dnn1_epochs = range(1, len(dnn1_loss)+1)
    dnn2_epochs = range(1, len(dnn2_loss)+1)
    dnn3_epochs = range(1, len(dnn3_loss)+1)
    dnn4_epochs = range(1, len(dnn4_loss)+1)
    dnn5_epochs = range(1, len(dnn5_loss)+1)
    plt.plot(lr_epochs, lr_loss, label="Linear Regression")
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

def main_test():
    """
    all 5 models
    but we only use train.csv
    and we split it into training, testing, and validations sets
    but validation only needed for LR input bc it doesn't do it itself
    the DNNs do it themselves
    """

    data = pd.read_csv("train.csv")
    data = preprocess(data)
    data = data.reindex(np.random.permutation(data.index)) # shuffle
    train_ids, x, y = separate_labels_ids(data, Y_HEADER)
    x = scale_inputs(x)
    y = scale_labels(y)
    x_train_lr, y_train_lr, x_test, y_test, x_val_lr, y_val_lr = train_test_val_split(x, y, 0.2)
    x_train = np.append(x_train_lr, x_val_lr, axis=0)
    y_train = np.append(y_train_lr, y_val_lr, axis=0)

    lr_learning_rate=1e-2
    lr_epochs = 200 # yes i know the parameter in the constructor is called early_stop not epochs, i don't want to change it
    lr_num_features = x_train_lr.shape[1]
    lr_sample_size = x_train_lr.shape[0]
    lr = LinearRegression(lr_learning_rate, lr_epochs, lr_num_features, lr_sample_size)
    lr_loss = lr.train(x_train_lr, y_train_lr, x_val_lr, y_val_lr)
    # predictions
    lr_bias, lr_var = bias_variance(lr, x_train, y_train, x_test, y_test)
    predictions = lr.test(x_test)
    lr_r2 = lr.R2(predictions, y_test)

    """
    https://www.geeksforgeeks.org/machine-learning/bias-vs-variance-in-machine-learning/
    """

    # DNN 8 out
    layers = [8]
    loss = 'mse'
    learning_rate=1e-2
    metrics = [keras.metrics.R2Score(), keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    epochs = 200
    batch_size = 32
    
    dnn1 = DNN(layers, loss, learning_rate, metrics, epochs, batch_size)
    dnn1.train(x_train, y_train)

    dnn1_bias, dnn1_var = bias_variance(dnn1.model, x_train, y_train, x_test, y_test)
    dnn1_l, dnn1_r2, dnn1_mse, dnn1_mae = dnn1.model.evaluate(x_test, y_test)

    
    # DNN 16 8 out
    layers = [16, 8]
    learning_rate = 1e-2
    metrics = [keras.metrics.R2Score(), keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    epochs = 200
    batch_size = 32
    
    dnn2 = DNN(layers, loss, learning_rate, metrics, epochs, batch_size)
    dnn2.train(x_train, y_train)

    dnn2_bias, dnn2_var = bias_variance(dnn2.model, x_train, y_train, x_test, y_test)
    dnn2_l, dnn2_r2, dnn2_mse, dnn2_mae = dnn2.model.evaluate(x_test, y_test)


    # DNN 16 8 4 out
    layers = [16, 8, 4]
    learning_rate = 1e-2
    metrics = [keras.metrics.R2Score(), keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    epochs = 200
    batch_size = 32
    
    dnn3 = DNN(layers, loss, learning_rate, metrics, epochs, batch_size)
    dnn3.train(x_train, y_train)

    
    dnn3_bias, dnn3_var = bias_variance(dnn3.model, x_train, y_train, x_test, y_test)
    dnn3_l, dnn3_r2, dnn3_mse, dnn3_mae = dnn3.model.evaluate(x_test, y_test)


    # DNN 30 16 8 4 out
    layers = [30, 16, 8, 4]
    learning_rate = 1e-2
    metrics = [keras.metrics.R2Score(), keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    epochs = 200
    batch_size = 32
    
    dnn4 = DNN(layers, loss, learning_rate, metrics, epochs, batch_size)
    dnn4.train(x_train, y_train)

    
    dnn4_bias, dnn4_var = bias_variance(dnn4.model, x_train, y_train, x_test, y_test)
    dnn4_l, dnn4_r2, dnn4_mse, dnn4_mae = dnn4.model.evaluate(x_test, y_test)


    # DNN 8 8 out
    layers = [8, 8]
    learning_rate = 1e-2
    metrics = [keras.metrics.R2Score(), keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    epochs = 200
    batch_size = 32
    
    dnn5 = DNN(layers, loss, learning_rate, metrics, epochs, batch_size)
    dnn5.train(x_train, y_train)

    
    dnn5_bias, dnn5_var = bias_variance(dnn5.model, x_train, y_train, x_test, y_test)
    dnn5_l, dnn5_r2, dnn5_mse, dnn5_mae = dnn5.model.evaluate(x_test, y_test)


    """
    underfit: high bias = oversimplifies, low variance, bad at testing/training
    overfit: high var = too sensitive to small changes, low bias, bad at testing
    want both to be moderate
    """
    print("\tLR\tDNN1\tDNN2\tDNN3\tDNN4\tDNN5")
    print(f"BIAS:\t{lr_bias:.2f}\t{dnn1_bias:.2f}\t{dnn2_bias:.2f}\t{dnn3_bias:.2f}\t{dnn4_bias:.2f}\t{dnn5_bias:.2f}")
    print(f"VAR:\t{lr_var:.2f}\t{dnn1_var:.2f}\t{dnn2_var:.2f}\t{dnn3_var:.2f}\t{dnn4_var:.2f}\t{dnn5_var:.2f}")
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

def bias_variance(model, x_train, y_train, x_test, y_test, runs=30):
    """
    copied and modified from gfg
    https://www.geeksforgeeks.org/machine-learning/bias-vs-variance-in-machine-learning/
    very similar to the one claude recommended
    """

    preds = []
    n = x_train.shape[0]
    for _ in range(runs):
        idx = np.random.choice(n, n, replace=True)
        x_sample = x_train[idx]
        y_sample = y_train[idx]
        model.fit(x_sample, y_sample)
        preds.append(model.predict(x_test))
    
    preds = np.array(preds)
    y_pred_mean = preds.mean(axis=0)
    
    bias_sq = ((y_test - y_pred_mean)**2).mean()
    variance = preds.var(axis=0).mean()
    total_error = bias_sq + variance
    
    return bias_sq, variance#, total_error

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
break for dinner and class
- *add my own model and resubmit pictures - 505pm
- *re-answer the data questions and make them better
- AI report 
    - *scale features
    - *fix DNN
    - *fix R2
    - *fix median age
    - *update bias/variance 
    - *smaller issues + after the previous fixes
wow that made things a lot better

10/1/26 start 1133am 
*check if i can still submit through kaggle - yes i can, late submission
*send email, about if kaggle needs to be reopened?
- *continue chatting with AI to find something that was wrong - maybe ask more targeted questions
    lets do chatgpt, and ask about interpretation?
- *update report - include pre and post AI
- *best model weights
- *function for best model to output submission 
129pm
- *clean up code
- *submit to kaggle - 341pm
- *submit report and weights and anything else 

"""


if __name__ == "__main__":
    main()