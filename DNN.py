


"""
https://www.geeksforgeeks.org/deep-learning/neural-networks-a-beginners-guide/
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = {
    'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
    'feature2': [0.5, 0.4, 0.3, 0.2, 0.1],
    'label': [0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)
X = df[['feature1', 'feature2']].values
y = df['label'].values

model = Sequential()
model.add(Dense(8, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

sequential is the newtork, dense is the layer
8 i think is the number of nodes

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
loss, optimizer, and metrics

model.fit(X, y, epochs=100, batch_size=1, verbose=1)

test_data = np.array([[0.2, 0.4]])
prediction = model.predict(test_data)
predicted_label = (prediction > 0.5).astype(int)

what were the specs that she outlined?
sgd for the optimizer and mse for the loss
8 - output
16 - 8 - output
16 - 8 - 4 - output
30 - 16 - 8 - 4 - output


design
DNN(layers=[30,16,8,4], loss='mse', optimizer='sgd', metrics='', epochs, batch_size)
train(x, y)
test(x)

how do i use a validation set for a Sequential?
see the early stopping i think
what are the Sequential.compile() options?
'sgd'
'meansquarederror'
'r2score'
'accuracy'
'precision'

how do i use tensorflow
model.summary() to see a summary of the lyaers

things to do later
maybe use EarlyStopping
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from Preprocessing import preprocess, train_test_val_split, train_val_split, separate_labels, remove_outliers

class DNN:
    def __init__(self, layers, loss, learning_rate, metrics, epochs, batch_size, early_stop=True):
        self.layers = layers
        self.loss = loss
        self.metrics = metrics # TODO remove?
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = keras.Sequential()
        self.init_layers()

        self.model.compile(
            loss=loss,
            optimizer=keras.optimizers.SGD(
                learning_rate = learning_rate,
                momentum=0.9
            ), 
            metrics=metrics
        )
        if early_stop:
            self.early_stop = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15, # if not improve for 3 epochs
                restore_best_weights=True,
                verbose=1,
            )
        else:
            self.early_stop = None
        return 

    def init_layers(self):
        for layer in self.layers:
            # self.model.add(keras.layers.Dense(layer, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(l1=1e-3, l2=1e-3)))
            self.model.add(keras.layers.Dense(layer, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3)))
            self.model.add(keras.layers.Dropout(0.2))
            # what is input_dim
        self.model.add(keras.layers.Dense(1))
        # print(f"DNN: init_layers: model summary:")
        # self.model.summary()
        return 

    def train(self, x, y):
        # verbose 0 - no
        # 1 - progress bar
        # 2 - one line per epoch
        if self.early_stop:
            self.history = self.model.fit(
                x, 
                y, 
                validation_split=0.2,
                epochs=self.epochs,
                batch_size=self.batch_size, 
                verbose=2,
                callbacks=[self.early_stop]
            )
            
        else:
            self.history = self.model.fit(
                x, 
                y, 
                validation_split=0.2,
                epochs=self.epochs,
                batch_size=self.batch_size, 
                verbose=2
            )
        self.model.summary()
        return

    def test(self, test_x: pd.DataFrame):
        """
        test_x is dataframe with id, features, and we want to predict the y
        what's the output?
        for now, just an array for ids and a parallel array for predictions

        splits the ids away, drops the ids from teh input, predicts based on the input df, 
        and returns parallel arrays ids and predictions
        """
        # print("\n\n\n\nTEST")
        ids = test_x['id'].to_numpy()
        test_x = test_x.drop(columns=['id'])
        predictions = np.array(self.model.predict(test_x)) # this returns an array
        predictions = np.reshape(predictions, (1, predictions.shape[0]))[0]
        # now they're both [12,3,4,,54,5]
        return ids, predictions

Y_HEADER = "TARGET_deathRate"

if __name__ == "__main__":

    # x_train = np.array([[1,2,3],[4,5,6],[10,11,12], [16,17,18], [13,14,15], [7,8,9]])
    # y_train = np.array([[28],[64],[136],[208], [172],[100]])
    # x_test = np.array([[19,20,21],[22,23,24]])
    # y_test = np.array([[244], [280]])
    # x_train = np.array([[1,1], [2,2], [3,3], [5,5], [6,6], [7,7], [8,8], [9,9], [10,10]])
    # y_train = np.array([[0],[0],[0],[0],[1],[1],[1],[1],[1]])
    # x_test = np.array([[4,4],[11,11]])
    # y_test = np.array([[0],[1]])

    train = pd.read_csv("train.csv")
    train = preprocess(train)
    # train = remove_outliers(train)
    x, y = separate_labels(train, Y_HEADER)
    # print(f"X: {x}, y: {y}")
    # x_train, y_train, x_test, y_test, x_val, y_val = train_test_val_split(x, y, .2)
    x_train, y_train, x_test, y_test = train_val_split(x, y, .2)
    test = pd.read_csv("test.csv")
    test = preprocess(test)

    layers = [8]
    loss = 'mse'
    metrics = [keras.metrics.R2Score(), keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()]
    epochs = 20
    batch_size = 32
    # learning_rate = 1e-12
    learning_rate=1e-4

    """
    r2 .5 - lr 1e-4, b_s 32, epochs 20, normalization layer, early stop, no sigmoid, just an 8 layer

    remove_outliers v1
    301684 - 20, 32, 1e-8
    """

    dnn1 = DNN(layers, loss, learning_rate, metrics, epochs, batch_size, early_stop=True)
    dnn1.train(x_train, y_train)
    loss, r2, mse, mae = dnn1.model.evaluate(x_test, y_test)
    print(f"\n\nTEST\nloss {loss}\nr2 {r2}\nmse {mse}\nmae {mae}")
    dnn1.test(test)
    # print(x_train.shape)
    # loss, acc = dnn1.model.evaluate(x_test, y_test)
    # print(f"loss {loss}, acc {acc}")
    # dnn1.test(x_test)

    # model = keras.Sequential()
    # model.add(keras.layers.Dense(8, activation="relu", name="layer1"))
    # model.summary()
    # x = tf.ones((1, 2))
    # y = model(x)
    # model.summary()