from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

dataAV = np.genfromtxt('xAV.txt')
dataV = np.genfromtxt('xV.txt')
dataAV0 = np.empty([1, 100])
dataAV1 = np.empty([1, 100])
dataAV2 = np.empty([1, 100])
counter = 0
for i in dataAV:
    dataAV0[0][counter] = i[0]
    dataAV1[0][counter] = i[1]
    dataAV2[0][counter] = i[2]
    counter += 1
dataAV0 = dataAV0[0]
dataAV1 = dataAV1[0]
dataAV2 = dataAV2[0]
# plt.hist(dataAV2, bins=50)
# plt.show()
visMean = dataV.mean()
visStd = dataV.std()
AV0mean = dataAV0.mean()
AV0std = dataAV0.std()
AV1mean = dataAV1.mean()
AV1std = dataAV1.std()
AV2mean = dataAV2.mean()
AV2std = dataAV2.std()
# dataAV1Mean = dataAV1.mean()
# dataAV2Mean = dataAV2.mean()
audioStd0 = sqrt((pow(visStd, 2) * pow(AV0std, 2)) / (pow(visStd, 2) - pow(AV0std, 2)))
audioMean0 = (AV0mean * (pow(audioStd0, 2) + pow(visStd, 2)) - pow(audioStd0, 2) * visMean) / pow(visStd, 2)

audioStd1 = sqrt((pow(visStd, 2) * pow(AV1std, 2)) / (pow(visStd, 2) - pow(AV1std, 2)))
audioMean1 = (AV1mean * (pow(audioStd1, 2) + pow(visStd, 2)) - pow(audioStd1, 2) * visMean) / pow(visStd, 2)

audioStd2 = sqrt((pow(visStd, 2) * pow(AV2std, 2)) / (pow(visStd, 2) - pow(AV2std, 2)))
audioMean2 = (AV2mean * (pow(audioStd2, 2) + pow(visStd, 2)) - pow(audioStd2, 2) * visMean) / pow(visStd, 2)
pass
