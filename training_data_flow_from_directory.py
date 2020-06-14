# Author: Blake Edwards
import os
import re
import sys

import math
import numpy as np
import pandas as pd
from sys import exit
from glob import glob
from dateutil import parser
from datetime import datetime

if len(sys.argv) != 2:
    print("[INCORRECT USAGE] try $ python3 /path/to/downloaded/dataset>")
    exit()
collected_data_path = sys.argv[1]

np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt;

plt.rcdefaults()

shift = 15
window_size = 45
mean_diff_threshold = 20000
training_samples = []
training_targets = []

classes = glob(os.path.join(collected_data_path, "*"))
for i in range(0, len(classes)):
    classes[i] = re.sub(str(collected_data_path) + str('/'), "", classes[i])
classes = sorted(classes, key=str.lower)
print(classes)
dustbin_label_loc = classes.index("dustbin")

processed = np.zeros(len(classes))
failed = np.zeros(len(classes))
for i in range(0, len(classes)):
    curr_data_path = os.path.join(collected_data_path, classes[i])
    print(f"processing  class: {classes[i]}")
    csvs = glob(os.path.join(curr_data_path, "*.csv"))
    for k in range(0, len(csvs)):
        print(f"processing {csvs[k]}")
        raw_data = pd.read_csv(csvs[k])
        try:
            raw_numpy = raw_data[["accelerometerTimestamp_sinceReboot(s)", "accelerometerAccelerationX(G)", "accelerometerAccelerationY(G)",
                "accelerometerAccelerationZ(G)"]].to_numpy()
        except:
            print("[ERROR] Loaded data contains incorrect columnn names")
            continue
        # todo remove redundant loops
        for j in range(0, len(raw_numpy)):
            raw_numpy[j][0] = raw_numpy[j][0] * 1000
        # print(raw_numpy)
        for j in range(30, len(raw_numpy) - (window_size + 30), 15):
            sample = raw_numpy[j:j + window_size]
            sample_diffs = np.diff(sample[:, 0], n=1, axis=0)
            std = np.std(sample_diffs, axis=0)
            mean = np.mean(sample_diffs, axis=0)
            diff = abs((1000 / 15) - mean)
            idx = classes.index(classes[i])
            if diff < mean_diff_threshold:
                # print(f"Processing sample with mean: {mean}")
                processed[idx] += 1
                training_samples.append(sample[:, 1:])
                target = np.zeros(len(classes))
                target[classes.index(classes[i])] = 1
                training_targets.append(target)
                # print(sample_diffs)
                # plt.subplot(211)
                # plt.title(f'{classes[i]} | Milliseconds between samples')
                # plt.plot(np.arange(1, window_size), sample_diffs)
                # plt.ylim([min(sample_diffs)-5, max(sample_diffs)+5])
                # plt.subplot(212)
                # plt.title("Accelerometer sample window")
                # plt.plot(np.arange(1, window_size+1), raw_data['accelerometerAccelerationX(G)'][j:j + window_size], 'r')
                # plt.plot(np.arange(1, window_size + 1), raw_data['accelerometerAccelerationY(G)'][j:j + window_size], 'g')
                # plt.plot(np.arange(1, window_size + 1), raw_data['accelerometerAccelerationZ(G)'][j:j + window_size], 'b')
                # max_y = max(max(raw_data['accelerometerAccelerationX(G)'][j:j + window_size]), max(raw_data['accelerometerAccelerationY(G)'][j:j + window_size]), max(raw_data['accelerometerAccelerationZ(G)'][j:j + window_size]))
                # min_y = min(min(raw_data['accelerometerAccelerationX(G)'][j:j + window_size]), min(raw_data['accelerometerAccelerationY(G)'][j:j + window_size]), min(raw_data['accelerometerAccelerationZ(G)'][j:j + window_size]))
                # plt.ylim([min_y-2, max_y+2])
                # plt.show()
                # plt.savefig(f'samples/{classes[i]}_{processed[idx]}.png')
                # plt.subplot(212).cla()
                # plt.subplot(211).cla()
            else:
                print(f"Dropping sample with mean: {mean}")
                failed[idx] += 1
                # print(sample_diffs)
                # plt.subplot(211)
                # plt.title(f'{classes[i]} | Milliseconds between samples')
                # plt.plot(np.arange(1, window_size), sample_diffs)
                # plt.ylim([min(sample_diffs)-100, max(sample_diffs)+100])
                # plt.subplot(212)
                # plt.title("Accelerometer sample window")
                # plt.plot(np.arange(1, window_size+1), raw_data['accelerometerAccelerationX(G)'][j:j + window_size], 'r')
                # plt.plot(np.arange(1, window_size + 1), raw_data['accelerometerAccelerationY(G)'][j:j + window_size], 'g')
                # plt.plot(np.arange(1, window_size + 1), raw_data['accelerometerAccelerationZ(G)'][j:j + window_size], 'b')
                # plt.ylim([-4.75, 4.75])
                # plt.show()

# saving training data
training_samples = np.asarray(training_samples)
training_targets = np.asarray(training_targets)

# create set of dustbin samples
num_dustbin_samples = 1000
dustbin_samples = np.random.uniform(-3, 3, size=(num_dustbin_samples, 45, 3))
# dustbin_samples = np.concatenate((dustbin_samples,  np.random.uniform(-2, 2, size=(num_dustbin_samples, 45, 3))))
# dustbin_samples = np.concatenate((dustbin_samples,  np.random.uniform(-1, 1, size=(num_dustbin_samples, 45, 3))))
dustbin_labels = np.zeros((len(dustbin_samples), len(classes)))
dustbin_labels[:, dustbin_label_loc] = 1
# print(dustbin_labels)

# generate random noise for dust bin class
train_x = np.concatenate((training_samples, dustbin_samples))
train_y = np.concatenate((training_targets, dustbin_labels))

np.save("training_samples", np.asarray(train_x))
np.save("training_targets", np.asarray(train_y))

print(f'\nClasses: {classes}')
print(f'Successful: {processed}')
print(f'Failed: {failed}')

print(f'\nSuccesfully processed {(sum(processed)) / float((sum(failed) + sum(processed))) * 100}% of the dataset')
print(f'Total samples: {sum(failed) + sum(processed)}')
print(f'Failed samples: {sum(failed)}')
print(f'Successful samples: {sum(processed)}')

print(f'\nClasses: {classes} \nSamples: {np.sum(train_y, axis=0)}')
