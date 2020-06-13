# Wearable Data Analysis
Apple Watch data engine framework and algorithms for human activity recognition using an accelerometer data stream.

## Table of Contents
- [Start Data Engine](#data-engine)
- [Visualized Data Samples](#visualized-data-samples)
- [Human Activity Recognition with Convolutional Neural Network](#human-activity-recognition)
- [Human Activity Recognition Dataset](https://bit.ly/3g4Ize1)

## Data Engine

### 
```console
$ python /raw_data_to_train_set/training_data_flow_from_directory.py
```
Sample output:
```
processing /debug_raw_data/left-hand/walk/stream Blake’s Apple Watch 200516 12_10_43.csv
.
. (processes all CSVs in target directories' sub-directories)
.
Successfully processed 99.97824825986079% of the dataset
Total samples: 27584.0
Failed samples: 6.0
Successful samples: 27578.0

Classes: ['drive', 'dustbin', 'lay', 'run', 'sit', 'stair', 'stand', 'walk'] 
[6076. 1004. 3572. 2315. 8374.  456.  344. 6437.]
```

## Visualized Data Samples
![sample figure](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/drive_sample.png)

![sample figure](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/lay_sample.png)

![sample figure](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/run_sample.png)

![sample figure](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/sit_sample.png)

![sample figure](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/stair_sample.png)

![sample figure](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/walk_sample.png)


## Human Activity Recognition

### Convolutional Neural Network (CNN)
#### Achieved 98.3% top-1 and 100% top-3 accuracy on test dataset


#### Architecture:
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

#### CNN Results:
```
Epoch 100/100
28578/28578 [==============================] - 5s 165us/sample - loss: 0.1019 - accuracy: 0.9627 - val_loss: 0.0806 - val_accuracy: 0.9698
loss: 0.08055452822212895
accuracy: 0.969767
['drive', 'dustbin', 'lay', 'run', 'sit', 'stair', 'stand', 'walk']
[[0.95424621 0.         0.00740619 0.00049375 0.03752469 0. 0.00016458 0.00016458]
 [0.         0.99601594 0.00099602 0.         0.00298805 0. 0.         0.        ]
 [0.01427772 0.         0.89277716 0.         0.09294513 0. 0.         0.        ]
 [0.         0.         0.         1.         0.         0. 0.         0.        ]
 [0.00680678 0.         0.01146406 0.         0.98172916 0. 0.         0.        ]
 [0.         0.         0.         0.         0.         1. 0.         0.        ]
 [0.00290698 0.         0.         0.         0.         0. 0.95930233 0.0377907 ]
 [0.00357309 0.         0.00015535 0.         0.00046606 0. 0.00077676 0.99502874]]
```
Confusion matrix:<br/>
![Confusion Matrix](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/confusion-matrix2.png)

## References

[Cole, C.A., Janos, B., Anshari, D., Thrasher, J.F., Strayer, S.M., & Valafar, H. (2020). Recognition of Smoking Gesture Using Smart Watch Technology. ArXiv, abs/2003.02735.](https://arxiv.org/pdf/2003.02735.pdf)

[Twomey, N.; Diethe, T.; Fafoutis, X.; Elsts, A.; McConville, R.; Flach, P.; Craddock, I. A Comprehensive Study of Activity Recognition Using Accelerometers. Informatics 2018, 5, 27.](https://www.mdpi.com/2227-9709/5/2/27)
