# Wearable-Data-Analysis
Apple Watch data engine framework and algorithms for human activity recognition
<br/>TODO: down sampling and processing of raw accelerometer data from activities

[Download Human Activity Recognition Datasets](https://drive.google.com/drive/folders/1DHQ8qPfv5xpGWKWLE99JIlssQM9yqHzg?usp=sharing)

Output from /raw_data_to_train_set/training_data_flow_from_directory.py :
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


References

[Cole, C.A., Janos, B., Anshari, D., Thrasher, J.F., Strayer, S.M., & Valafar, H. (2020). Recognition of Smoking Gesture Using Smart Watch Technology. ArXiv, abs/2003.02735.](https://arxiv.org/pdf/2003.02735.pdf)

[Twomey, N.; Diethe, T.; Fafoutis, X.; Elsts, A.; McConville, R.; Flach, P.; Craddock, I. A Comprehensive Study of Activity Recognition Using Accelerometers. Informatics 2018, 5, 27.](https://www.mdpi.com/2227-9709/5/2/27)
