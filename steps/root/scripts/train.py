import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
from azureml.core import Run
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='Test data folder mounting point')
parser.add_argument('--epochs', type=str, dest='epochs', help='Amount of epochs to train')
parser.add_argument('--batch_size', type=str, dest='batch_size', help='Batch size')
parser.add_argument('--model_name', type=str, dest='model_name', help='Model name')
args = parser.parse_args()

data_folder = args.data_folder
print('Data folder:', data_folder)

dataset_train = np.load(os.path.join(data_folder, 'dataset_train.npy'))
dataset_test = np.load(os.path.join(data_folder, 'dataset_test.npy'))
dataset_train = pd.DataFrame(dataset_train)
dataset_test = pd.DataFrame(dataset_test)

run = Run.get_context()

batch_size = int(args.batch_size)
epochs = int(args.epochs)

# Building the autoencoder
from tensorflow.keras import metrics

autoencoder = Sequential()
#Decode
autoencoder.add(InputLayer((3952,)))
autoencoder.add(Dense(1000, activation= 'relu' ))
#Bottleneck
autoencoder.add(Dense(120, activation= 'relu' ))
#Encode
autoencoder.add(Dense(1000, activation= 'relu' ))

autoencoder.add(Dense(3952, activation= 'sigmoid' ))
autoencoder.summary()


# Training the autoencoder with a custom loss function
def custom_loss(y_true,y_pred):
    y_mask=tf.keras.backend.clip(y_true, 0, 0.01)*100
    return K.mean(K.square(y_mask*(y_pred - y_true)), axis=-1)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

autoencoder.compile(loss=custom_loss, optimizer='adam')

autoencoder.fit(np.array(dataset_train),
                np.array(dataset_train),
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                callbacks=[reduce_lr,early_stopping_callback])

os.makedirs('outputs', exist_ok=True)
autoencoder.save(f"outputs/{str(args.model_name)}")