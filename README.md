[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
# Wearable Data Analysis
Apple Watch data engine framework, dataset, and algorithms for human activity recognition.

## Table of Contents
- [Get Started](#get-started)
- [Run Data Engine](#data-engine)
- [Visualized Data Samples](#visualized-data-samples)
- [Human Activity Recognition with Convolutional Neural Network](#human-activity-recognition)
- [Human Activity Recognition Dataset](https://bit.ly/3g4Ize1)
- [References](#references)

## Get Started
```console
$ git clone git@github.com:blakete/Wearable-Data-Analysis.git
```

## Run Data Engine
```console
$ python3 training_data_flow_from_directory.py /path/to/dataset
processing raw dataset
.
. (processes all CSVs in target directories' sub-directories)
.
Successfully processed 100.0% of the dataset
Total samples: 61990.0
Failed samples: 0.0
Successful samples: 61990.0

Classes: ['drive', 'dustbin', 'lay', 'run', 'sit', 'skate', 'stair', 'stand', 'walk'] 
Samples: [15141.  1004. 15477.  8261. 11659.   185.   532.  3877.  6854.]
```

## Visualized Data Samples
![sample figure](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/drive_sample.png)

![sample figure](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/run_sample.png)

![sample figure](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/walk_sample.png)

![sample figure](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/stair_sample.png)


## Human Activity Recognition
### Achieved 95.9% top-1 and 100% top-3 accuracy on test dataset
### Convolutional Neural Network (CNN) Architecture:
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
#### Example CNN Results:
```
Epoch 150/150
62273/62273 [==============================] - 12s 196us/sample - val_loss: 0.1124 - val_accuracy: 0.9593
loss: 0.11244082237378554
accuracy: 0.95932424
```
Confusion matrix:<br/>
![Confusion Matrix](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/confusion-matrix2.png)

## Requirements
* \>= Apple A12 or \>= iPhone XR 
* \>= Apple Watch Series 3
* [Wearable CoPilot App (coming soon!)](https://bit.ly/3g9T9Ak)

## References

[Cole, C.A., Janos, B., Anshari, D., Thrasher, J.F., Strayer, S.M., & Valafar, H. (2020). Recognition of Smoking Gesture Using Smart Watch Technology. ArXiv, abs/2003.02735.](https://arxiv.org/pdf/2003.02735.pdf)

[Twomey, N.; Diethe, T.; Fafoutis, X.; Elsts, A.; McConville, R.; Flach, P.; Craddock, I. A Comprehensive Study of Activity Recognition Using Accelerometers. Informatics 2018, 5, 27.](https://www.mdpi.com/2227-9709/5/2/27)
