#-*- coding: utf-8 -*-
import os
import numpy as np

def convert(sequence, index):
    f = open(os.path.join(sequence+'.txt'), 'r')
    kitti_data = f.readlines()

    ## Convert : [r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3] -> [x, y, z, pitch, yaw, roll]

    input = kitti_data[int(index)]
    input = input.split()
    input = [float(d) for d in input]

    T = np.array([input[3], input[7], input[11]])
    Rmat = np.eye(3)
    Rmat[0] = input[0:3]
    Rmat[1] = input[4:7]
    Rmat[2] = input[8:11]

    ### We must minus first translate
    # first_input = kitti_data[0]
    # first_input = first_input.split()
    # first_input = [float(d) for d in first_input]
    # first_T = np.array([first_input[3], first_input[7], first_input[11]])
    # cam_pose = list(T - first_T)

    cam_pose = list(T)
    ### z : forwarad, x : right, y : down
    pitch = np.arctan(Rmat[1, 0] / Rmat[0, 0]) # pitch
    yaw = np.arctan(-Rmat[2, 0] / np.sqrt(np.square(Rmat[2, 1]) + np.square(Rmat[2, 2]))) # yaw
    roll = np.arctan(Rmat[2, 1] / Rmat[2, 2]) # roll

    cam_pose.append(pitch) # gamma
    cam_pose.append(yaw) # beta
    cam_pose.append(roll) # alpha


    # ### Cykim ver2
    # cam_pose = list(T)
    # Zc = np.array([0, 0, 1])
    # Z0 = np.matmul(Rmat, Zc)
    # Yc = np.array([0, 1, 0])
    # Xc = np.array([1, 0, 0])
    # Y0 = np.matmul(Rmat, Yc)
    # X0 = np.matmul(Rmat, Xc)
    # Zc_Y0 = np.inner(Zc, Y0)
    #
    # pitch = np.arcsin(max(-1, min(1, Zc_Y0)))
    #
    # Zc_XZ0 = Zc - np.inner(Zc, Y0) * Y0 / np.linalg.norm(Y0)
    # Zc_XZ0 = Zc_XZ0 / np.linalg.norm(Zc_XZ0)
    # Zc_XZ0_Z0l = np.inner(Zc_XZ0, Z0)
    # Zc_XZ0_X0 = np.inner(Zc, X0)
    # if np.arcsin(Zc_XZ0_X0) >= 0:
    #     yaw = np.arccos(max(-1, min(1, Zc_XZ0_Z0l)))
    # else:
    #     yaw = -np.arccos(max(-1, min(1, Zc_XZ0_Z0l)))
    #
    # Xc_Y0 = np.inner(Xc, Y0)
    # roll = np.arcsin(max(-1, min(1, Xc_Y0)))
    #
    # cam_pose.append(pitch)
    # cam_pose.append(yaw)
    # cam_pose.append(roll)

    return cam_pose