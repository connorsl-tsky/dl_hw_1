


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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import R2Score
import tensorflow as tf


class DNN:
    def __init__(self, layers, loss, optimizer, metrics, epochs, batch_size):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics # TODO remove?
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = tf.keras.models.Sequential()
        self.init_layers()

        self.model.compile(loss=loss,
              optimizer=optimizer, metrics=metrics)
        return 

    def init_layers(self):
        for layer in self.layers:
            self.model.add(Dense(layer, activation='relu'))
            # what is input_dim
        self.model.add(Dense(1, activation='sigmoid'))
        print(f"DNN: init_layers: model summary: {self.model.summary()}")
        return 

    def train(self, x, y):
        # verbose 0 - no
        # 1 - progress bar
        # 2 - one line per epoch
        self.model.fit(x, y, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        return

    def test(self, test_x):
        print(f"DNN: test: prediction: {self.model.predict(test_x)}")
        return

if __name__ == "__main__":

    x_train = np.array([[1,2,3],[4,5,6],[10,11,12], [16,17,18], [13,14,15], [7,8,9]])
    y_train = np.array([[28],[64],[136],[208], [172],[100]])
    x_test = np.array([[19,20,21],[22,23,24]])
    y_test = np.array([[244], [280]])

    layers = [8]
    loss = 'mse'
    optimizer = 'sgd'
    metrics = [tf.keras.metrics.R2Score(), tf.keras.metrics.Accuracy(), 'precision']
    epochs = 10
    batch_size = 10

    dnn1 = DNN(layers, loss, optimizer, metrics, epochs, batch_size)
    dnn1.train(x_train, y_train)
    dnn1.test(x_test)