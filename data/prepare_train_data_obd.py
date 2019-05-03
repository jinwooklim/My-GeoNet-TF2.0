#-*- coding: utf-8 -*-
import os
import argparse
import math
import numpy as np


def create_obd(pose_dir, dump_dir):
    '''
    KITTI data pose -> obd parameter
    '''
    L_car = 2.7
    cam_car = (math.pi / 180.0) * 90.0
    angle_scale = 540.0 / 35.0
    R_car = np.zeros((2, 2))
    R_car[0, 0], R_car[0, 1], R_car[1, 0], R_car[1, 1] = math.cos(cam_car), math.sin(cam_car), -math.sin(
        cam_car), math.cos(cam_car)
    bias_steering = 0.0
    fps = 9.64764
    alpha = 1.0 / (3.6 * fps)  # alpha = 1.0 / fps / 3.6
    obd_frame_1max = 0.0
    temp1 = 0.0
    temp2 = 0.0
    s_temp1 = 0.0
    s_temp2 = 0.0

    for pose_file in (os.listdir(os.path.join(pose_dir, "poses"))):
        if int(os.path.splitext(pose_file)[0]) > 10:
            break
        x = []
        y = []
        z = []
        pitch = []
        yaw = []
        roll = []
        obd_frame_prev = [[], []]
        obd_frame = [[], []]

        print("create obd : ", os.path.join(pose_dir, "poses", pose_file)) # **.txt
        if not os.path.exists(os.path.join(dump_dir, 'virtual_obd')):
            os.mkdir(os.path.join(dump_dir, 'virtual_obd'))
        wf = open(os.path.join(dump_dir, 'virtual_obd', pose_file), 'w')

        with open(os.path.join(pose_dir, "poses", pose_file), 'r') as rf:
            data_list = rf.readlines()

        # for i in range(len(data_list)-1): # number of frames - 1
        for i in range(len(data_list)):
            data = data_list[i].split()

            T = np.array([data[3], data[7], data[11]])
            Rmat = np.eye(3)
            Rmat[0] = data[0:3]
            Rmat[1] = data[4:7]
            Rmat[2] = data[8:11]

            cam_pose = list(T)
            alpha_z = np.arctan(Rmat[1, 0] / Rmat[0, 0])  # z-axis
            beta_y = np.arctan(-Rmat[2, 0] / np.sqrt(np.square(Rmat[2, 1]) + np.square(Rmat[2, 2])))  # y-axis
            gamma_x = np.arctan(Rmat[2, 1] / Rmat[2, 2])  # x-axis
            cam_pose.append(gamma_x)
            cam_pose.append(-beta_y)
            cam_pose.append(-alpha_z)

            data = cam_pose

            x.append(float(data[0]))
            y.append(float(data[1]))
            z.append(float(data[2]))
            pitch.append(float(data[3]))
            yaw.append(float(data[4]))
            roll.append(float(data[5]))

            if i != 0:
                tx = x[i] - x[i-1]
                tz = z[i] - z[i-1]
                theta = yaw[i] - yaw[i-1]

                if tz == 0.0:
                    tz = 0.000001
                if abs(theta) < 0.000001:
                    theta = 0.000001

                s = theta * math.sqrt(tx ** 2 + tz ** 2) / (2.0 * math.sin(theta / 2.0))
                if s < 0.0001:
                    s = 0.0001
                a = L_car / s * theta

                obd_frame[1] = int(round(bias_steering + a / np.pi * 180.0 * angle_scale))

                obd_frame[0] = int(round(s / alpha))

                if i != 1:
                    if abs(obd_frame[1] - temp1) > 4.0 * angle_scale or abs(obd_frame[0]) < 4.0:
                        if obd_frame[1] - temp1 != 0.0:
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

                if i != 1:
                    if abs(obd_frame[0] - s_temp1) > 10.0:
                        if obd_frame[0] - s_temp1 != 0.0:
                            obd_frame[0] = s_temp1 + abs(s_temp1 - s_temp2) * (obd_frame[0] - s_temp1) / abs(obd_frame[0] - s_temp1)
                        else:
                            obd_frame[0] = s_temp1
                        s_temp2 = s_temp1
                        s_temp1 = obd_frame[0]
                    else:
                        s_temp2 = s_temp1
                        s_temp1 = obd_frame[0]
                else:
                    s_temp1 = obd_frame[0]

                if abs(obd_frame[1]) < 70.0 * angle_scale:
                    obd_frame_1max = max(obd_frame_1max, abs(obd_frame[1]))

                wf.write('%f %f\n' % (obd_frame[0], obd_frame[1]))

                # if i != 1:
                #     with open(os.path.join(dump_dir, os.path.splitext(pose_file)[0], '%06d_obd.txt'%i), 'w') as wf:
                #         wf.write('%f,%f,%f,%f\n' % (obd_frame_prev[0], obd_frame_prev[1], obd_frame[0], obd_frame[1]))
                # else:
                #     with open(os.path.join(dump_dir, os.path.splitext(pose_file)[0], '%06d_obd.txt'%i), 'w') as wf:
                #         wf.write('%f,%f,%f,%f\n' % (obd_frame[0], obd_frame[1], obd_frame[0], obd_frame[1]))
                # obd_frame_prev = obd_frame

    for obd_file in (os.listdir(os.path.join(dump_dir, 'virtual_obd'))):
        with open(os.path.join(dump_dir, 'virtual_obd', obd_file), 'r') as rf:
            data_list = rf.readlines()
            for i in range(len(data_list)):
                data = data_list[i].split(' ')
                if i == 0:
                    prev_data = data
                elif i >= 1:
                    with open(os.path.join(dump_dir, os.path.splitext(obd_file)[0], '%06d_obd.txt'%i), 'w') as wf:
                        wf.write('%f,%f,%f,%f' % (float(prev_data[0]), float(prev_data[1]), float(data[0]), float(data[1])))
                prev_data = data


if __name__ == "__main__":
    create_obd("D:\PROJECT\smartcar\KITTI_odometry\dataset", "./virtual_obd")
