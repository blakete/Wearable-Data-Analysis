# Author: Blake Edwards
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras.datasets
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, Conv2D, MaxPooling2D

epochs = 100
batch_size = 32
num_classes = 8
img_rows, img_cols = 45, 3

# load generated samples
classes = ['drive', 'dustbin', 'lay', 'run', 'sit', 'stair', 'stand', 'walk']
# train_x = np.load("../raw_data_to_train_set/training_samples.npy")
# train_y = np.load("../raw_data_to_train_set/training_targets.npy")
train_x = np.load("/Users/blakeedwards/Desktop/Repos/research/Wearable-Data-Analysis/train_set/training_samples.npy")
train_y = np.load("/Users/blakeedwards/Desktop/Repos/research/Wearable-Data-Analysis/train_set/training_targets.npy")

checkpoint =  ModelCheckpoint("checkpoint.hdf5", save_best_only=True, monitor='val_accuracy')
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, min_delta=0.003, min_lr=0.00001)
callbacks = [checkpoint,reduce_lr_on_plateau]

# todo: split off test set when we have larger dataset
(x_train, y_train), (x_test, y_test) = (train_x, train_y), (train_x, train_y)

x_train = x_train.astype('float16')
x_test = x_test.astype('float16')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train.reshape(-1, 45, 3, 1)
x_test = x_test.reshape(-1, 45, 3, 1)

input_shape = (45, 3, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(num_classes, activation='softmax'))
# SparseCategoricalCrossentropy
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(lr=0.05),
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=callbacks)
score = model.evaluate(x_test, y_test, verbose=0)
print('loss:', score[0])
print('accuracy:', score[1])

np.set_printoptions(suppress=True)
y_pred = model.predict(x_test)
y_pred = [np.argmax(y) for y in y_pred]
y_test_max = [np.argmax(y) for y in y_test]

cf_matrix = confusion_matrix(y_test_max, y_pred, normalize='true')

sns.heatmap(cf_matrix, annot=True)
plt.title(f"{classes}")
plt.show()

print(classes)
np.set_printoptions(suppress=True)
print(cf_matrix)































