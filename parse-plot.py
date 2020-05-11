
import csv
import numpy as np
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

from numpy import genfromtxt

import csv
import numpy as np
from datetime import datetime
import matplotlib as mpl

from numpy import genfromtxt

times = []
accelerometer = []
aX = []
aY = []
aZ = []

vX = []
vY = []
vZ = []

with open("../Left hand/Driving car/driving_2.csv") as file:
	csvObj = csv.reader(file, delimiter=',')
	cnt = 0
	for row in csvObj:
		if cnt > 0:
			row[0] = datetime.strptime(row[0][0:-6], '%Y-%m-%d %H:%M:%S.%f').timestamp() * 1000
		cnt += 1
		times.append(row)
		x=0
		try:
			x = float(row[11])
		except:
			print(row[2])
		y=0
		try:
			y = float(row[12])
		except:
			print(row[3])
		z=0
		try:
			z = float(row[13])
		except:
			print(row[4])
		aX.append(x)
		aY.append(y)
		aZ.append(z)

timestamps = []
for row in times:
	timestamps.append(row[0])

dT = []
obj3 = []
obj4 = []
start = 240000
allTimes = []
size = 5000

# moveAvg2 = diff = times[start][0] - times[start-1][0]
beta = 0.95
for j in range(2, len(times)):
	diff = times[j][0] - times[j-1][0]
	dT.append(diff)
	if(diff > 5000):
		break
	allTimes.append(times[j][0])
	vX.append(diff*aX[j])
	vY.append(diff*aY[j])
	vZ.append(diff*aZ[j])

movingAvgX = vX[0]
movingAvgY = vY[0]
movingAvgZ = vZ[0]

avgsX = []
avgsY = []
avgsZ = []

for j in range(0, len(vX)):
	avgX = movingAvgX*beta + vX[j]*(1-beta)
	avgY = movingAvgY*beta + vY[j]*(1-beta)
	avgZ = movingAvgZ*beta + vZ[j]*(1-beta)
	avgsX.append(avgX + 10)
	avgsY.append(avgY + 7)
	avgsZ.append(avgZ + 80.6)

'''
with open('output.csv', 'w') as csvfile:
	cwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	for row in obj1:
		cwriter.writerow(row)
'''

print("done")

low = 1
high = 2000

plt.figure(0) 
plt.plot(allTimes[low:high], avgsX[low:high], "r")
plt.plot(allTimes[low:high], avgsY[low:high], "g")
plt.plot(allTimes[low:high], avgsZ[low:high], "b")
plt.ylabel('acceleration')
plt.title("moving average")

plt.figure(1) 
plt.plot(timestamps[low:high], aX[low:high], "r")
plt.plot(timestamps[low:high], aY[low:high], "g")
plt.plot(timestamps[low:high], aZ[low:high], "b")
plt.ylabel('acceleration')
plt.title("raw data")

plt.figure(2) 
plt.plot(timestamps[low:high], dT[low:high], "r")

		
plt.show()
