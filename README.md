# Wearable-Data-Analysis
Apple Watch data engine framework and algorithms for human activity recognition using an accelerometer data stream.

[Download Human Activity Recognition Datasets](https://drive.google.com/drive/folders/1DHQ8qPfv5xpGWKWLE99JIlssQM9yqHzg?usp=sharing)

## Human Activity Recognition Solutions

### Convolutional Neural Network
Results:
```
Epoch 30/30
5931/5931 [==============================] - 0s 67us/sample - accuracy: 0.9391 - val_accuracy: 0.9412
Test loss: 0.2481557119758952
Test accuracy: 0.9411566
Confusion matrix:
['dustbin', 'run', 'stair', 'walk']
[[ 497    0    0    7]
 [   0  153    0  124]
 [   0    0    0  160]
 [  23   35    0 4932]]
```

## Data Engine

### $ python /raw_data_to_train_set/training_data_flow_from_directory.py :
```
...
Processing /debug_raw_data/left-hand/walk/stream Blake’s Apple Watch 191022 19_43_36.csv
Dropping sample with mean: 119.7367386364124
Dropping sample with mean: 119.73890719698234
Processing /debug_raw_data/left-hand/walk/stream Blake’s Apple Watch 200516 12_10_43.csv

Classes: ['run', 'stairs', 'walk']
Successful: [316, 35, 5124]
Failed: [21, 0, 88]

Successfully processed 98.04799426934098% of the dataset
Total samples: 5584
Failed samples: 109
Successful samples: 5475
```

![sample figure](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/run_34.png)
![sample figure](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/stair_16.png)
![sample figure](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/walk_1668.png)
![sample figure](https://github.com/blakete/Wearable-Data-Analysis/blob/master/figures/class_distribution.png)
Failed samples have a sampling rate outside the acceptable range of 10 Hz - 25 Hz
<br/>TODO: down sampling of raw accelerometer data from activities


## References

[Cole, C.A., Janos, B., Anshari, D., Thrasher, J.F., Strayer, S.M., & Valafar, H. (2020). Recognition of Smoking Gesture Using Smart Watch Technology. ArXiv, abs/2003.02735.](https://arxiv.org/pdf/2003.02735.pdf)

[Twomey, N.; Diethe, T.; Fafoutis, X.; Elsts, A.; McConville, R.; Flach, P.; Craddock, I. A Comprehensive Study of Activity Recognition Using Accelerometers. Informatics 2018, 5, 27.](https://www.mdpi.com/2227-9709/5/2/27)
