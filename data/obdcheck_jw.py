import os
import math
import numpy as np

fps = 9.64764  # 10.0
alpha = 1.0 / (3.6 * fps)
beta = 0.000901
gamma = 0.0
delta = 0.05 * math.pi / 180.0 * -0.0062
L_car = 2.7
bias_steering = 0.0
angle_scale = 540.0 / 35.0

seq = "09"
f = open('D:\\PROJECT\\smartcar\\dump_pose\\virtual_obd\\%s.txt'%seq,'rb')
data = f.readlines()
obdSpeedList = []
obdAngleList = []
for i in range(len(data)):
    datum = data[i].split()
    speed = float(datum[0])
    angle = float(datum[1])
    obdSpeedList.append(speed)
    obdAngleList.append(angle)

# Plot
with open("./obd2pose.txt", 'w') as wf:
    for i in range(len(data)):
        a = obdAngleList[i] / angle_scale * np.pi / 180.0 + bias_steering
        s = obdSpeedList[i] * alpha

        theta = obdAngleList[i] / angle_scale / L_car * s * np.pi / 180.0

        if theta < 0.000001:
            theta = 0.000001

        phy = 0.0
        tx = -2.0 / theta * np.sin(theta / 2.0) * s * np.sin(a + theta / 2.0 + phy)
        tz = 2.0 / theta * np.sin(theta / 2.0) * s * np.cos(a + theta / 2.0 + phy)

        wf.write("%s %s %s %s %s %s\n"%(tx, 0.0, tz, 0.0, theta, 0.0))