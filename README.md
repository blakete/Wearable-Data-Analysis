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
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 43, 1, 16)         160       
_________________________________________________________________
flatten (Flatten)            (None, 688)               0         
_________________________________________________________________
dense (Dense)                (None, 32)                22048     
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 132       
=================================================================
Total params: 22,340
Trainable params: 22,340
Non-trainable params: 0
```

CNN Results:
```
Epoch 30/30
6050/6050 [==============================] - 0s 74us/sample - loss: 0.2519 - accuracy: 0.9235 - val_loss: 0.2491 - val_accuracy: 0.9245
loss: 0.24911313154480674
accuracy: 0.9244628
['dustbin', 'run', 'stair', 'walk']
[[ 499    0    0    5]
 [   1  333    0   62]
 [   0    1    0  159]
 [  55  174    0 4761]]
```
Updated 05/20/2020
![Confusion Matrix](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/confusion-matrix.png)

## References

[Cole, C.A., Janos, B., Anshari, D., Thrasher, J.F., Strayer, S.M., & Valafar, H. (2020). Recognition of Smoking Gesture Using Smart Watch Technology. ArXiv, abs/2003.02735.](https://arxiv.org/pdf/2003.02735.pdf)

[Twomey, N.; Diethe, T.; Fafoutis, X.; Elsts, A.; McConville, R.; Flach, P.; Craddock, I. A Comprehensive Study of Activity Recognition Using Accelerometers. Informatics 2018, 5, 27.](https://www.mdpi.com/2227-9709/5/2/27)
