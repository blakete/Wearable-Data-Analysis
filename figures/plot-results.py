
# Author: Blake Edwards
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/blakeedwards/Desktop/eskd_test_results.csv')

four_four_teu = df['4_4_teu'].to_numpy()
six_four_teu = df['6_4_teu'].to_numpy()
eight_four_teu = df['8_4_teu'].to_numpy()

four_four_teu = four_four_teu[np.logical_not(np.isnan(four_four_teu))]
six_four_teu = six_four_teu[np.logical_not(np.isnan(six_four_teu))]
eight_four_teu = eight_four_teu[np.logical_not(np.isnan(eight_four_teu))]

four_four_test = df['4_4_testaccuracy'].to_numpy()
six_four_test = df['6_4_testaccuracy'].to_numpy()
eight_four_test = df['8_4_testaccuracy'].to_numpy()

four_four_test = four_four_test[np.logical_not(np.isnan(four_four_test))]
six_four_test = six_four_test[np.logical_not(np.isnan(six_four_test))]
eight_four_test = eight_four_test[np.logical_not(np.isnan(eight_four_test))]

plt.figure(1)
plt.plot(four_four_teu, four_four_test, c='r')
plt.plot(six_four_teu, six_four_test, c='g')
plt.plot(eight_four_teu, eight_four_test, c='b')

# plot baseline accuracies
base_testaccs = df['base_testacc'].to_numpy()
base_trainaccs = df['base_trainacc'].to_numpy()

base_testaccs = base_testaccs[np.logical_not(np.isnan(base_testaccs))]
base_trainaccs = base_trainaccs[np.logical_not(np.isnan(base_trainaccs))]

mean_test = np.mean(base_testaccs)
mean_train = np.mean(base_trainaccs)
stdev_test = np.std(base_testaccs)
stdev_train = np.std(base_trainaccs)

largest_teu = 200
baseline_testacc_y  = np.empty(largest_teu)
baseline_testacc_y.fill(mean_test)
baseline_trainacc_y  = np.empty(largest_teu)
baseline_trainacc_y.fill(mean_train)
baseline_x = np.arange(0,largest_teu, 1)

plt.plot(baseline_x, baseline_testacc_y)
plt.ylim((0.40,0.55))
plt.savefig("student_testacc.png")


df_teacher = pd.read_csv('/Users/blakeedwards/Desktop/teacher_results.csv')

four_te = df_teacher['4_teacher_epoch'].to_numpy()
six_te = df_teacher['6_teacher_epoch'].to_numpy()
eight_te = df_teacher['8_teacher_epoch'].to_numpy()
four_noDA_te = df_teacher['4_noDA_teacher_epoch'].to_numpy()
six_noDA_te = df_teacher['6_noDA_teacher_epoch'].to_numpy()
four_te = four_te[np.logical_not(np.isnan(four_te))]
six_te = six_te[np.logical_not(np.isnan(six_te))]
eight_te = eight_te[np.logical_not(np.isnan(eight_te))]
four_noDA_te = four_noDA_te[np.logical_not(np.isnan(four_noDA_te))]
six_noDA_te = six_noDA_te[np.logical_not(np.isnan(six_noDA_te))]

four_t_testacc = df_teacher['4_teacher_testacc'].to_numpy()
six_t_testacc = df_teacher['6_teacher_testacc'].to_numpy()
eight_t_testacc = df_teacher['8_teacher_testacc'].to_numpy()
four_noDA_testacc = df_teacher['4_noDA_teacher_testacc']
six_noDA_testacc = df_teacher['6_noDA_teacher_testacc']
four_t_testacc = four_t_testacc[np.logical_not(np.isnan(four_t_testacc))]
six_t_testacc = six_t_testacc[np.logical_not(np.isnan(six_t_testacc))]
eight_t_testacc = eight_t_testacc[np.logical_not(np.isnan(eight_t_testacc))]
four_noDA_testacc = four_noDA_testacc[np.logical_not(np.isnan(four_noDA_testacc))]
six_noDA_testacc = six_noDA_testacc[np.logical_not(np.isnan(six_noDA_testacc))]

# pre-compute sorting
four_sort = four_te.argsort()
six_sort = six_te.argsort()
eight_sort = eight_te.argsort()
four_noDA_sort = four_noDA_te.argsort()
six_noDA_sort = six_noDA_te.argsort()

plt.figure(2)
plt.plot(four_te[four_sort], four_t_testacc[four_sort], c='r')
plt.plot(six_te[six_sort], six_t_testacc[six_sort], c='g')
plt.plot(eight_te[eight_sort], eight_t_testacc[eight_sort], c='b')
plt.plot(four_noDA_te[four_noDA_sort], four_noDA_testacc[four_noDA_sort], c='r')
plt.plot(six_noDA_te[six_noDA_sort], six_noDA_testacc[six_noDA_sort], c='g')
plt.savefig("teacher_testacc.png")

plt.show()

# print(four_t_testacc[four_sort])



