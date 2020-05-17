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
mean_diff_threshold = 35
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
    processed = 0
    failed = 0
    for k in range(0, len(csvs)):
        print(f"processing {csvs[k]}")
        raw_data = pd.read_csv(csvs[k])
        raw_numpy = raw_data[["accelerometerTimestamp_sinceReboot(s)", "accelerometerAccelerationX(G)", "accelerometerAccelerationY(G)",
                              "accelerometerAccelerationZ(G)"]].to_numpy()
        # todo remove redundant loops
        for j in range(0, len(raw_numpy)):
            raw_numpy[j][0] = raw_numpy[j][0]*1000
        # print(raw_numpy)
        for j in range(0, len(raw_numpy) - window_size, 15):
            sample = raw_numpy[j:j + window_size]
            sample_diffs = np.diff(sample[:,0], n=1, axis=0)
            std = np.std(sample_diffs, axis=0)
            mean = np.mean(sample_diffs, axis=0)
            diff = abs((1000/15)-mean)
            if diff < mean_diff_threshold:
                print(f"Processing sample with mean: {mean}")
                processed = processed + 1
                training_samples.append(sample)
                target = np.zeros(3)
                target[classes.index(classes[i])] = 1
                training_targets.append(target)
            else:
                print(f"Dropping sample with mean: {mean}")
                failed = failed + 1
                # print(sample_diffs)
                # plt.subplot(211)
                # plt.title("Milliseconds between samples")
                # plt.plot(np.arange(1, window_size), sample_diffs)
                # plt.ylim([min(sample_diffs)-5, max(sample_diffs)+5])
                # plt.subplot(212)
                # plt.title("Accelerometer sample window")
                # plt.plot(np.arange(1, window_size+1), raw_data['accelerometerAccelerationX(G)'][j:j + window_size], 'r')
                # plt.plot(np.arange(1, window_size + 1), raw_data['accelerometerAccelerationY(G)'][j:j + window_size], 'g')
                # plt.plot(np.arange(1, window_size + 1), raw_data['accelerometerAccelerationZ(G)'][j:j + window_size], 'b')
                # plt.ylim([-3.5, 3.5])
                # plt.show()
print(f'\nSuccesfully processed {(processed)/float((failed+processed))*100}% of the dataset')
print(f'Total samples: {failed+processed}')
print(f'Failed samples: {failed}')
print(f'Successful samples: {processed}')


np.save("training_samples", np.asarray(training_samples))
np.save("training_targets", np.asarray(training_targets))










