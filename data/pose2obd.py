#-*- coding: utf-8 -*-
import numpy as np
import cv2
import math
import os
import scipy.special as sp
import random
#random.seed(8964)
#np.random.seed(8964)


def convert(seq_length, dump_dir, filename):
    seq_obd_frame = []
    
    # int(filename) == tgt_obd_idx
    tgt_obd_idx = int(filename)
    offset = int(seq_length/2)

    ## KITTI data pose -> obd parameter
    L_car = 2.7
    cam_car = (math.pi / 180.0) * 90.0
    angle_scale = 540.0 / 35.0
    R_car = np.zeros((2, 2))
    R_car[0, 0], R_car[0, 1], R_car[1, 0], R_car[1, 1] = math.cos(cam_car), math.sin(cam_car), -math.sin(
        cam_car), math.cos(cam_car)
    bias_steering = 0.0
    fps = 9.64764
    alpha = 1.0 / (3.6 * fps) # alpha = 1.0 / fps / 3.6
    beta = 0.000901
    gamma = 0.0
    delta = 0.05 * math.pi / 180.0 * -0.0062
    phy = 0.0
    obd_frame_1max = 0.0
    temp = 0.0
    temp1 = 0.0
    temp2 = 0.0
    s_temp1 = 0.0
    s_temp2 = 0.0
    obd_speed_min = 100.0
    barrior = 35.0 * np.pi / 180.0

    for pidx in range(tgt_obd_idx, tgt_obd_idx+offset*2):
        try:
            #print("in : ", tgt_obd_idx, tgt_obd_idx+offset*2, pidx)
            f1 = open(os.path.join(dump_dir, "%06d_pose.txt"%(pidx-1)))
            f2 = open(os.path.join(dump_dir, "%06d_pose.txt"%(pidx)))
            f1_data = f1.readlines()
            f2_data = f2.readlines()
            f1_data = f1_data[0].split(',')
            f2_data = f2_data[0].split(',')
            
            x1 = float(f1_data[0])
            z1 = float(f1_data[2])
            yaw1 = float(f1_data[4])
            x2 = float(f2_data[0])
            z2 = float(f2_data[2])
            yaw2 = float(f2_data[4])
        except IOError:
            print("No pose file : ", tgt_obd_idx, pidx)
            if pidx > tgt_obd_idx:
                f1 = open(os.path.join(dump_dir, "%06d"%(tgt_obd_idx - 1)+"_pose.txt"))
                f2 = open(os.path.join(dump_dir, "%06d"%(tgt_obd_idx - 1)+"_pose.txt"))
            else:
                f1 = open(os.path.join(dump_dir, "%06d"%(tgt_obd_idx)+"_pose.txt"))
                f2 = open(os.path.join(dump_dir, "%06d"%(tgt_obd_idx)+"_pose.txt"))
            f1_data = f1.readlines()
            f2_data = f2.readlines()
            f1_data = f1_data[0].split(',')
            f2_data = f2_data[0].split(',')
 
            x1 = float(f1_data[0])
            z1 = float(f1_data[2])
            yaw1 = float(f1_data[4])
            x2 = float(f2_data[0])
            z2 = float(f2_data[2])
            yaw2 = float(f2_data[4])

        ## Calculate obd
        tx = x2 - x1
        tz = z2 - z1
        theta = yaw2 - yaw1
        if tz == 0:
            tz = 0.000001
        if abs(theta) < 0.000001:
            theta = 0.000001

        phi = np.arctan(tx / tz) - \
              (1.0 / 2.0 + 2.0 * L_car * np.sin(theta / 2.0) / theta / np.sqrt(np.square(tx) + np.square(tz))) * theta
        d = np.sqrt(np.square(tx) + np.square(tz))
        s = theta * math.sqrt(tx ** 2 + tz ** 2) / (2 * math.sin(theta / 2))
        if s < 0.0001:
            s = 0.0001
        a = L_car / s * theta
    
        obd_speed_min = min(obd_speed_min, s)
        obd_frame = [[], []]
        obd_frame[1] = int(round(bias_steering + a / np.pi * 180.0 * angle_scale))
        obd_frame[0] = int(round(s / alpha))

        if tgt_obd_idx != 1:
            if abs(obd_frame[1] - temp1) > 4.0 * angle_scale or abs(obd_frame[0]) < 4.0:
                if(obd_frame[1] - temp1 != 0.0):
                    obd_frame[1] = temp1 + abs(temp1 - temp2) * (obd_frame[1] - temp1) / abs(obd_frame[1] - temp1)
                else:
                    obd_frame[1] = temp1
                temp2 = temp1
                temp1 = obd_frame[1]
            else:
                temp2 = temp1
                temp1 = obd_frame[1]
        else:
            temp1 = obd_frame[1]

        # if tgt_obd_idx != 1:
        #     if abs(obd_frame[0] - s_temp1) > 10.0:
        #         if obd_frame[0] - s_temp1 != 0.0:
        #             obd_frame[0] = s_temp1 + abs(s_temp1 - s_temp2) * (obd_frame[0] - s_temp1) / abs(
        #                 obd_frame[0] - s_temp1)
        #         else:
        #             obd_frame[0] = s_temp1
        #         s_temp2 = s_temp1
        #         s_temp1 = obd_frame[0]
        #     else:
        #         s_temp2 = s_temp1
        #         s_temp1 = obd_frame[0]
        # else:
        #     s_temp1 = obd_frame[0]

        if abs(obd_frame[1]) < 70.0 * angle_scale:
            obd_frame_1max = max(obd_frame_1max, abs(obd_frame[1]))

        seq_obd_frame.append(obd_frame)

    return seq_obd_frame, offset

