# Author: Blake Edwards
import os
import re
import math
import numpy as np
import pandas as pd
from glob import glob
from dateutil import parser
from datetime import datetime
import matplotlib.pyplot as plt

shift = 15
window_size = 45
mean_diff_threshold = 20
training_samples = []
training_targets = []


labels = np.arange(5)

collected_data_path = "/Users/blakeedwards/Desktop/Repos/research/Wearable-Data-Analysis/debug_raw_data/left-hand"


classes = glob(os.path.join(collected_data_path, "*"))
for i in range(0, len(classes)):
    classes[i] = re.sub(str(collected_data_path)+str('/'), "", classes[i])
classes = sorted(classes, key=str.lower)

for i in range(0, len(classes)):
    curr_data_path = os.path.join(collected_data_path, classes[i])
    print(f"processing  class: {classes[i]}")
    csvs = glob(os.path.join(curr_data_path, "*.csv"))
    for k in range(0, len(csvs)):
        print(f"processing {csvs[k]}")
        raw_data = pd.read_csv(csvs[k])
        raw_numpy = raw_data[["loggingTime(txt)", "accelerometerAccelerationX(G)", "accelerometerAccelerationY(G)",
                              "accelerometerAccelerationZ(G)"]].to_numpy()
        # todo remove redundant loops
        for j in range(0, len(raw_numpy)):
            raw_numpy[j][0] = parser.parse(raw_numpy[j][0]).timestamp() * 1000
        for j in range(0, len(raw_numpy) - window_size, 15):
            sample = raw_numpy[j:j + window_size]
            sample_diffs = np.diff(sample[:,0], n=1, axis=0)
            std = np.std(sample_diffs, axis=0)
            mean = np.mean(sample_diffs, axis=0)
            diff = abs((1000/15)-mean)
            if diff < mean_diff_threshold:
                training_samples.append(sample)
                target = np.zeros(3)
                target[classes.index(classes[i])] = 1
                training_targets.append(target)

np.save("training_samples", np.asarray(training_samples))
np.save("training_targets", np.asarray(training_targets))










