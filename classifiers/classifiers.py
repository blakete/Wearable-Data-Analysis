# Author: Blake Edwards
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.keras.datasets
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten

epochs = 50
batch_size = 32
num_classes = 6
dustbin_label_loc = 0
img_rows, img_cols = 45, 3

# load generated samples
classes = ['dustbin', 'lay', 'run', 'sit', 'stair', 'walk']
train_x = np.load("../raw_data_to_train_set/training_samples.npy")
train_y = np.load("../raw_data_to_train_set/training_targets.npy")
train_x = train_x[:,:,1:]
print(train_x.shape)

# create set of dustbin samples
num_dustbin_samples = 500
dustbin_samples = np.random.uniform(-3,3,size=(num_dustbin_samples,45,3))
dustbin_labels = np.zeros((num_dustbin_samples, num_classes))
dustbin_labels[:, dustbin_label_loc] = 1

# generate random noise for dust bin class
train_x = np.concatenate((train_x, dustbin_samples))
train_y = np.concatenate((train_y, dustbin_labels))

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
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adadelta(lr=0.01),
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('loss:', score[0])
print('accuracy:', score[1])

y_pred = model.predict(x_test)
y_pred = [np.argmax(y) for y in y_pred]
y_test_max = [np.argmax(y) for y in y_test]

cf_matrix = confusion_matrix(y_test_max, y_pred)

sns.heatmap(cf_matrix, annot=True)
plt.title(f"{classes}")
plt.show()

print(classes)
print(cf_matrix)































