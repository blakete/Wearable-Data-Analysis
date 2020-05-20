# Wearable-Data-Analysis
Apple Watch data engine framework and algorithms for human activity recognition using an accelerometer data stream.

[Download Human Activity Recognition Dataset](https://drive.google.com/drive/folders/1sIqmj5eS39np8gXwqByge_Rd_xmH5Efr?usp=sharing)

## Data Engine

### $ python /raw_data_to_train_set/training_data_flow_from_directory.py :
```
...
processing /debug_raw_data/left-hand/walk/stream Blake’s Apple Watch 200516 12_10_43.csv

Classes: ['drive', 'dustbin', 'lay', 'run', 'sit', 'stair', 'walk']
Successful: [1.465e+03 4.000e+00 1.180e+03 4.790e+02 4.227e+03 1.600e+02 5.282e+03]
Failed: [0. 0. 0. 0. 0. 0. 0.]

Succesfully processed 100.0% of the dataset
Total samples: 12797.0
Failed samples: 0.0
Successful samples: 12797.0
```
![sample figure](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/classes.png)

## Visualized Data Samples
![sample figure](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/run_34.png)
![sample figure](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/walk_1668.png)

<br/>TODO: down sampling of raw accelerometer data from activities

## Human Activity Recognition

### Convolutional Neural Network (CNN)

Architecture:
```_________________________________________________________________
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 43, 1, 32)         320       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 42, 1, 64)         4160      
_________________________________________________________________
batch_normalization (BatchNo (None, 42, 1, 64)         256       
_________________________________________________________________
flatten (Flatten)            (None, 2688)              0         
_________________________________________________________________
dense (Dense)                (None, 64)                172096    
_________________________________________________________________
batch_normalization_1 (Batch (None, 64)                256       
_________________________________________________________________
activation (Activation)      (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080      
_________________________________________________________________
batch_normalization_2 (Batch (None, 32)                128       
_________________________________________________________________
activation_1 (Activation)    (None, 32)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 7)                 231       
=================================================================
Total params: 179,527
Trainable params: 179,207
Non-trainable params: 320
```

CNN Results:
Updated 05/20/2020
```
Epoch 50/50
13797/13797 [==============================] - 2s 164us/sample - loss: 0.2066 - accuracy: 0.9305 - val_loss: 0.2226 - val_accuracy: 0.9218
loss: 0.22263002375008806
accuracy: 0.9217946
Confusion matrix:
['drive', 'dustbin', 'lay', 'run', 'sit', 'stair', 'walk']
[[0.86962457 0.         0.0668942  0.02798635 0.01569966 0.        0.01979522]
 [0.         0.99203187 0.         0.00099602 0.00398406 0.        0.00298805]
 [0.02966102 0.         0.91864407 0.         0.05169492 0.        0.        ]
 [0.02713987 0.         0.         0.97286013 0.         0.        0.        ]
 [0.0508635  0.         0.09770523 0.         0.85095813 0.        0.00047315]
 [0.         0.         0.         0.         0.         0.71875   0.28125   ]
 [0.01344188 0.         0.00359712 0.00018932 0.00037864 0.00056797 0.98182507]]
```
![Confusion Matrix](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/confusion-matrix.png)

## References

[Cole, C.A., Janos, B., Anshari, D., Thrasher, J.F., Strayer, S.M., & Valafar, H. (2020). Recognition of Smoking Gesture Using Smart Watch Technology. ArXiv, abs/2003.02735.](https://arxiv.org/pdf/2003.02735.pdf)

[Twomey, N.; Diethe, T.; Fafoutis, X.; Elsts, A.; McConville, R.; Flach, P.; Craddock, I. A Comprehensive Study of Activity Recognition Using Accelerometers. Informatics 2018, 5, 27.](https://www.mdpi.com/2227-9709/5/2/27)
